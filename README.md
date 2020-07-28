# RandomForest
## Final project Machine learning course
Alex Klushnikov, Benjamin Ahr

# שימוש
- יש להתקין פייתון פרסה 3.6 ומעלה 
- להעתיק את הפרוייקט על ידי הפעולה הבאה:
~~~
      $git clone https://github.com/alexklus/RandomForest.git
~~~
- להתקין את הספרים מהקובץ הדרישות 
 ~~~
      pip install requirements.txt
~~~
## הוראות הרצה:
~~~
1) open main.py file
2) manage RandomForest object as you wish
        example:

         forest = RandomForest("mushrooms", #dataSet name
                          n_boostrap=100, #number of row to consider in each random tree (max = n-1)
                          n_features=24, #number of features to consider in each random tree
                          test_size=0.2, #test dataset size(float = percentage,int = number of rows )
                          n_trees=20, #number of trees in forest
                          tree_max_depth=10 #max tree depth etc max number of questions to ask
                          )
3) run forest.test_model() to test your model
###Optional
forest.print_forest() #print each decision tree in forest

if you want to predict new examples with the same features without label
run make_new_prediction(file_path,#path to file with data
                            forest #trained model
                            )
##Add new data set
- add csv file to Data directory
- edit config.ini file

[dataSetName]
path = ./Data/"dataSetName".csv
label = #name of the feature to classify
unnecessary_features = #features to drop from data set if there is no feature equal None

Example:
[Titanic]
path = ./Data/Titanic.csv
label = Survived
unnecessary_features = PassengerId, Name, Ticket, Cabin

~~~