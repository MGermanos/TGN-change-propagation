from datetime import datetime
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
# import tensorflow as tf
# gpus = tf.config.list_physical_devices('GPU')


def intersection(ar1,ar2,start_time,end_time): # to compute the common commits between two files
  # start time is added so that we start counting from the commit where the newest file was added in
  # end time is added so that we don't count the commits that are in the train set
  ar3=[]
  for i in ar2:
    if i in ar1 and i not in ar3 and i>=start_time and i <= end_time:
      ar3.append(i)
  return len(ar3)

def union(ar1,ar2,start_time,end_time): # to compute all the commits that file A or file B were in
  ar3=[]
  for i in ar1:
    if i not in ar3 and i>= start_time and  i <= end_time:
      ar3.append(i)
  return len(ar3)

def cochangeability(fileA,fileB,end_time,changes_of_file): # measure the cochangeability of two files
  # compute the date the newest commit was added
  start_time=max(changes_of_file[fileA][0],changes_of_file[fileB][0])
  return intersection(changes_of_file[fileA],changes_of_file[fileB],start_time,end_time)/union(changes_of_file[fileA],changes_of_file[fileB],start_time,end_time)

#select the projects you would like to test
projects=["alamofire","ant","cassandra","cassandrawebsite","flutter","gephi","hbase","lucene","monitorcontrol","pydriller","react","rocketmqclients","spark","wwwsite"]
for project in projects:

    print(project)
    for repeating_shuffle in range(5):
          print(repeating_shuffle)
          f=open("ShuffledData/"+str(repeating_shuffle)+"/ChangeSets/"+project+".csv")
          changes=f.readlines()
          f.close()
        
          changes=[x.rstrip().rstrip(",").split(",") for x in changes] # parse the lines of file
        
          for i in range(len(changes)):
              changes[i][0]=datetime.strptime(changes[i][0][:-6],'%Y-%m-%d %H:%M:%S') # get time of commit
              
          temp=[]
          commits_we_discard=0
          average_size_of_commit=[]
        
          train_changes=changes[:int(len(changes)*0.5)] 
        
          for i in train_changes[1:]:
            average_size_of_commit.append(len(i))
          p90=np.percentile(average_size_of_commit, 90)
        
          for i in changes: # remove any changes that contain a single file, or contain more than 19 files
              if len(i)>2 and len(i)<p90:
                  temp.append(i)
              else:
                  commits_we_discard+=1
                  
          changes=temp
        
          # take a subset of data
          changes=changes[:1000] 
        
          train_changes=changes[:int(len(changes)*0.5)] # split into train subset
      
          lstm_layer_size_1=4
          lstm_layer_size_2=4
          cutoff_value_for_coch=0.005
          cutoff_change_set_predicted_size=95
          optimizer="adam"
          epochs=5
              
                
          # keeps track of all the changes a file was included in
          changes_of_file={}
          for i in range(len(train_changes)): # i is the number of commit
            for j in train_changes[i][1:]: # j is the name of the file
              if j not in changes_of_file:
                changes_of_file[j]=[]
              changes_of_file[j].append(i)
    
          all_files=list(changes_of_file.keys()) # keeps track of all the files we viewed so far
    
    
          # build graph
          temporal_node_changes=[]
          temporal_node_cochanges=[]
    
          # this is for the first commit.
          # all the files have a cochangeability of 1
          changed_nodes={}
          coch={}
          for i in train_changes[0][1:]:
            changed_nodes[i]=1
            for j in train_changes[0][1:]:
              if i ==j:
                continue
              if i not in coch:
                coch[i]={}
    
              coch[i][j]=1
              
          temporal_node_changes.append(changed_nodes)
          temporal_node_cochanges.append(coch)
    
          # now we loop through the remaining commits
          for i in range(1,len(train_changes)): # i is the number of the commit
            current_change=train_changes[i][1:]
    
            changed_nodes={}
            coch={}
            # copy the previous graph
            for j in temporal_node_changes[-1]: # go to the last saved dictionary, and loop through the keys
              changed_nodes[j]=0# add the keys to the current dictionary, but mark all of them as unchanged for this commit
              
            for j in temporal_node_cochanges[-1]: # loop through the co-change graph, and copy the dictionary of all the nodes
              coch[j]=dict(temporal_node_cochanges[-1][j])
    
            for j in train_changes[i][1:]: # mark all the changed files in this commit as changed in the new graph
              # this also handles the files that were added in this comit
              changed_nodes[j]=1
    
            temporal_node_changes.append(changed_nodes) # add the label network to the temporal graph
            
            for j in train_changes[i][1:]: # i is the commit number, j is a changed file, k is any previous file
              for k in temporal_node_changes[-1]:
    
                if j==k: # j and k are the same file
                  continue
    
                if j not in coch: # j was recently added, so we add it to coch graph
                  coch[j]={}
    
                c=cochangeability(fileA=j,fileB=k,end_time=i,changes_of_file=changes_of_file) # compute its co-changeability
                coch[j][k]=c
            temporal_node_cochanges.append(coch) # add the coch graph to temporal one
          # to keep track of metrics
          sensitivity=[]
          specificity=[]
          ppv=[]
          gmean=[]
          fmeasure=[]
          accuracy=[]
          mcc=[]
          auc=[]
          average_size_of_commit=[]
    
          for i in train_changes[1:]:
            average_size_of_commit.append(len(i))
          average_size_of_commit=np.percentile(average_size_of_commit, cutoff_change_set_predicted_size)
    
          # now we start looping through the test commits
          test_commits_passed=-1
          for test_commit_index in range(int(len(changes)*0.5),len(changes)): # test commit index is the number of the commit
              test_commits_passed+=1
              test_commit_index,len(changes)
              test_commit=changes[test_commit_index][1:] # test commit only contains the files of this commit
              change_set=[]
              queue=[]    
    
              i=0
              while test_commit[i] not in temporal_node_cochanges[-1]: # find the file to start the propagation from
                  i+=1
                  if i>=len(test_commit):
                      break
              if i >= len(test_commit):
                  continue
    
              queue.append(test_commit[i]) # start with a file
              change_set.append(test_commit[i])
    
              while len(queue)>0 and len(change_set)< average_size_of_commit:
                current=queue.pop()
                for neighbor in sorted(temporal_node_cochanges[-1][current]): # loop through the coch of current. neighbor is the file name
                  if temporal_node_cochanges[-1][current][neighbor]<cutoff_value_for_coch or neighbor in change_set or len(change_set)>= average_size_of_commit: 
                  # ignore very low coch neighbors
                  # and neighbors that are already in the change set
                    continue
    
                  # review the history of the two nodes
                  X={"coch":[]}
                  y={"isChanged":[]}
    
                  for go_back_index in range(1,len(temporal_node_cochanges)-1):
                    go_back=temporal_node_changes[go_back_index]
                    go_back_one_more_choch=temporal_node_cochanges[go_back_index-1]
                    if current in go_back and go_back[current]==1: # our current node changed at this time, go back one step and check if they were neighbors
                        if current in go_back_one_more_choch and neighbor in go_back_one_more_choch[current]:
                          X["coch"].append(go_back_one_more_choch[current][neighbor]) # add their cochangeability
                          y["isChanged"].append(go_back[neighbor])# and add if the file was changed
                          
                  # create dataframes
                  if len(X["coch"])==0:
                    continue
                  #reshape data to fit and test
                  X=np.array(X["coch"])
                  X = X.reshape(X.shape[0], 1, 1)
                  y=pd.DataFrame(y)
                  
                  X_test=np.array([temporal_node_cochanges[-1][current][neighbor]])
                  X_test = X_test.reshape(X_test.shape[0], 1, 1)
                  # create model + fit +predict
                  model = Sequential()
                  model.add(LSTM(lstm_layer_size_1, activation='tanh', return_sequences=True, input_shape=(1, 1)))
                  model.add(LSTM(lstm_layer_size_2, activation='tanh'))
                  model.add(Dense(1, activation='sigmoid'))
                  model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
                  model.fit(X,y,batch_size=len(X),  epochs=epochs,verbose=0)
                  prediction=model.predict(X_test,verbose=0)
    
                  # if the probability is higher than 0.5, the model predicted that the file will change, and we add it to the queue
                  if prediction[0][0]>0.5:
                    queue.append(neighbor)
                    change_set.append(neighbor)
                    
              actual_set=test_commit
              tp=0
              tn=0
              fp=0
              fn=0
              actual_set=test_commit
              for i in actual_set:
                  if i in change_set:
                      tp+=1
                  else:
                      fn+=1
              for i in change_set:
                  if i not in actual_set:
                      fp+=1
              tn=len(temporal_node_changes[-1])-(tp+fp+fn)
              
              results_file=open("Results/ConfMatrix/directed_"+project+"_results"+str(repeating_shuffle)+".csv","a+")
              results_file.write(str(tp)+","+str(tn)+","+str(fp)+","+str(fn)+"\n")
              results_file.close()
              
              sensitivity.append(tp/(tp+fn))
              specificity.append(tn/(tn+fp))
              ppv.append(tp/(tp+fp))
              gmean.append((sensitivity[-1]*specificity[-1])**(1/2))
              fmeasure.append((2*ppv[-1]*sensitivity[-1])/(ppv[-1]+sensitivity[-1]))
              accuracy.append((tp+tn)/(tp+tn+fp+fn))
              mcc.append((tp*tn-fp*fn)/(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**(1/2)))
              auc.append(0.5*((tp/(tp+fn)+(tn/(tn+fp)))))
              
              # print(test_commit_index,len(changes),np.mean(sensitivity),end=" ")
              # print(np.mean(specificity),end=" ")
              # print(np.mean(ppv),end=" ")
              # print(np.mean(auc))
    
              # update the last slice of the temporal network
              for changed_node in actual_set:
                if changed_node not in changes_of_file:
                  changes_of_file[changed_node]=[]
                changes_of_file[changed_node].append(test_commit_index)
    
              changed_nodes={}
              coch={}
              # copy the previous graph
              for j in temporal_node_changes[-1]: # go to the last saved dictionary, and loop through the keys
                changed_nodes[j]=0# add the keys to the current dictionary, but mark all of them as unchanged for this commit
                
              for j in temporal_node_cochanges[-1]: # loop through the co-change graph, and copy the dictionary of all the nodes
                coch[j]=dict(temporal_node_cochanges[-1][j])
    
              for j in actual_set: # mark all the changed files in this commit as changed in the new graph
                # this also handles the files that were added in this comit
                changed_nodes[j]=1
    
              temporal_node_changes.append(changed_nodes) # add the label network to the temporal graph
              
              for j in actual_set: # i is the commit number, j is a changed file, k is any previous file
                for k in temporal_node_changes[-1]:
    
                  if j==k: # j and k are the same file
                    continue
    
                  if j not in coch: # j was recently added, so we add it to coch graph
                    coch[j]={}
    
                  c=cochangeability(fileA=j,fileB=k,end_time=test_commit_index,changes_of_file=changes_of_file) # compute its co-changeability
                  coch[j][k]=c
                  
              temporal_node_cochanges.append(coch) # add the coch graph to temporal one
          results_file=open("Results/Metrics/directed_"+project+"_results.csv","a+")
          results_file.write("directed,"+project+","+str(np.mean(sensitivity))+","+str(np.mean(specificity))+","+str(np.mean(ppv))+","+str(np.mean(gmean))+","+str(np.mean(fmeasure))+","+str(np.mean(accuracy))+","+str(np.mean(mcc))+","+str(np.mean(auc))+","+str(lstm_layer_size_1)+","+str(lstm_layer_size_2)+","+str(cutoff_value_for_coch)+","+str(cutoff_change_set_predicted_size)+","+optimizer+","+str(epochs)+"\n")
          results_file.close()
          
          results_file=open("Results/ConfMatrix/undirected_"+project+"_results.csv","a+")
          results_file.write("***\n")
          results_file.close()
