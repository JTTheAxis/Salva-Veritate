#-----------------------------------------------------------------------------
# Name:        Salva Veritate
# Purpose:     To create statistical models and use them to verify the validity
# 					of various news items.                            
# Author:      Jason Tian
# Created:     22-Mar-2020
# Updated:     19-Jun-2020 
#-----------------------------------------------------------------------------

import sys, csv, os 
import inspect
#Note: to run this program, numpy, pandas and sklearn are all required modules. As well, if your compiler cannot find the required packages, insert the path
#to said packages in the same place that I have done so below. 
sys.path.insert(0, r"C:\Users\JXYTi\AppData\Local\Programs\Python\Python38\site-packages")
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from newsitem import NewsItem
from article import Article
from report import Report


fakeList=["Extremely", "Believe", "Trick", "Great"]
archive=[]

os.chdir("Files")

#Animals Article
news_handle=open("Wild Animals.txt", "r")
news=news_handle.readlines()
news_handle.close()

for i in range(len(news)-1):
    if news[i]!="?":
        news[i]=news[i][:-1]
    
animals=NewsItem(title=news[0], date=news[1], org=news[2], author=news[3], valid=True, content=" ".join(news[4:]))
animals.read()
archive.append(animals)

#Inauguration
news_handle=open("Inauguration.txt", "r")
news=news_handle.readlines()
news_handle.close()

for i in range(len(news)-1):
    if news[i]!="?":
        news[i]=news[i][:-1]
    
trump=NewsItem(title=news[0], date=news[1], org=news[2], author=news[3], valid=True, content=" ".join(news[4:]))
trump.read()
archive.append(trump)

#Nova Scotia
news_handle=open("Scotian Tragedy.txt", "r")
news=news_handle.readlines()
news_handle.close()

for i in range(len(news)-1):
    if news[i]!="?":
        news[i]=news[i][:-1]
    
scotia=NewsItem(title=news[0], date=news[1], org=news[2], author=news[3], valid=news[4], content=" ".join(news[5:]))
scotia.read()
archive.append(scotia)

for work in archive:
    filename="'" + str(work.title) + "'" + " Analysis.txt"
    news_data=open(filename, "w")
    news_data.write("Title: " + work.title + "\n")
    news_data.write("Date Published: " + work.date + "\n")
    news_data.write("Author: " + work.author + "\n")
    news_data.write("Word Count: " + str(len(work.content)) + "\n")
    work.keywordCount=work.countKeywords(fakeList)
    news_data.write("Invalidity Ratio: " + str(work.dubRatio()) + "\n")
    news_data.close()

#Article class test
news_handle=open("Anti-Hillary.txt", "r", encoding="utf8")
news=news_handle.readlines()
news_handle.close()

for i in range(len(news)-1):
    if news[i]!="?":
        news[i]=news[i][:-1]
        
fear=Article(title=news[0], date=news[1], org=news[2], author=news[3], valid=False, content=" ".join(news[4:]), bias=True, stance="Against", target="Hillary Clinton")
fear.read()
print(fear.countKeywords(fakeList))
fear.argument()

#Report class test
news_handle=open("Anti-Trump.txt", "r", encoding="utf8")
news=news_handle.readlines()
news_handle.close()

for i in range(len(news)-1):
    if news[i]!="?":
        news[i]=news[i][:-1]

revolt=Report(title=news[0], date=news[1], org=news[2], author=news[3], valid=True, content=" ".join(news[4:]), accuracy=0.90)
revolt.read()
print(revolt.countKeywords(fakeList))
if revolt.credibilityCheck():
    print("The article '" + revolt.title + "' is credible.")
else:
    print("The article '" + revolt.title + "'is not credible.")

#Data Type Limitations
#Byte: Consists of an integer number of bits, used to encode a single character of text in the computer. Can only be comprised of bits.
#Integer: An integral data type, represents mathematical integer values. May be restricted from negative values depending on whether or not the integer is "signed". Is comprised of a series of bits, cannot take in non-integer values such as alphabetic strings or symbols. Limited in value based on how many bits are used to form the datum, maximum integer size is 2^n-1, where n is the number of bits. Commonly used number of bits are 8, 16, 32, and 64.  
#Float: Also comprised of a fixed amount of bits, like the byte and integer. Therefore, it has a fixed size. The number of bits used defines how "precise" the floating point value can be. If more bits are used, then the number of significant digits which are recorded can be much higher, at the cost of exponentially reduced speed. Commonly used number of bits are 32-bit, or "single precision", and 64-bit, or "double precision". However, floating point types will always suffer from a loss of accuracy due to there possibly being an infinite number of digits in certain real numbers. Cannot take in non-integer values.
#Double: 64-bit "double precision" value. Has similar limiations to the floating point type, but with a much higher possible number of significant digits. Cannot take in non-integer values.
#String: A sequence of alphanumeric characters and symbols. If fixed-length, then the limit of memory used is set as a constant. If variable-length, then the memory used changes based on the amount allocated and/or required, with the maximum limit being the memory capacity of the computer. Representable characters limited by the designated alphabet in the language being used.
#Boolean: A data type with two possible values: True or False. Cannot take any values other than these two, and can only take one of the two possible values at any given point in time. 
#Array: A sequence of values, which can, collectively, be any type, but must all be the same type as one another. Indices in an array should only be integers in a specific range (generally corresponding to the length of the array). Length of an array is constrained by the maximum available memory of the computer. 

test=["asdf", "wewr", "sfsdf", "fhjdsakfhsdlk", "zcvb", "sdfhjk", "asdhfjadlf"]

def bb_sort(array):
    '''
    Performs bubble sort on a list (in ascending order).
    
    Parameters
    ----------
    array : list
    	The list to perform bubble sort on.
    
    Returns
    -------
    list
    	The final list sorted in ascending order.
            
    '''    
    for term in array:
        for i in range(len(array)-1):
            if array[i] > array[i+1]: 
                array[i], array[i+1]=array[i+1], array[i]
    return array

#print(bb_sort(test))

def sel_sort(array):
    '''
    Performs selection sort on a list (in ascending order).
    
    Parameters
    ----------
    array : list
    	The list to perform selection sort on.
    
    Returns
    -------
    list
    	The final list sorted in ascending order.
            
    '''        
    for i in range(len(array)):
        minIndex=i
        for j in range(i+1, len(array)):
            if array[i]>array[j]:
                minIndex=j
        array[i], array[minIndex] = array[minIndex], array[i]
    return array

#print(sel_sort(test))

def lin_search(array, item):
    '''
    Uses linear search to look for an item in the provided list.
    
    Parameters
    ----------
    array : list
    	The list to search.
    item : str, int, float
    	The item to look for.
        
    Returns
    -------
    int
    	The index at which the desired item is contained. Is -1 if the item is not found.
            
    '''         
    for i in range(len(array)):
        if array[i]==item:
            return i
    return -1    

#print(lin_search(test, 1))

def bin_search(array, item):
    '''
    Uses binary search to look for an item in the provided list.
    Keeps a constant note of the key left and right endpoints, changing them as necessary as the binary search progresses.
    
    Parameters
    ----------
    array : list
    	The list to search.
    item : str, int, float
    	The item to look for.
        
    Returns
    -------
    int
    	The index at which the desired item is contained. Is -1 if the item is not found.
            
    '''          
    found=False
    index=-1
    l=0
    r=len(array)-1
    while l<=r and not found:
        m=(l+r)//2
        if array[m]==item:
            found=True
            index=m
        else:
            if item < array[m]:
                r=m-1
            elif item > array[m]:
                l=m+1
    return index

#print(bin_search(test, 'as'))

#Note for Mr. Seidel: Change the path below to whatever path leads to the right folder on your PC.
os.chdir(r"C:\Users\JXYTi\Desktop\Jason's files\School\Grade 12 (2019-2020)\Computer Science")
work_names=[]
#Parsing csv with full dataset
with open("News Dataset.csv", encoding="utf8") as csv_file:
    csv_reader=csv.reader(csv_file, delimiter=",")
    line=0
    for row in csv_reader:
        if line==0:
            line+=1
        else:
            work_names.append(row[1])

#Bubble Sort
#bubble=bb_sort(work_names)
#print("First Work:", bubble[0])
#print("Last Work:", bubble[-1])

#Selection Sort
#selection=sel_sort(work_names)
#for i in range(len(selection)):
#    print("Alphabetical Work no." + str(i+1) + ":", work_names[i])

#Linear Search    
#print(lin_search(bubble, "Lessons from Obama's deal with Iran"))

#Binary Search
#print(bin_search(selection, "Lessons from Obama's deal with Iran"))

#In this project, the PassiveAggressiveClassifier can be treated as the "model". 

while True:
    print("Welcome to Salva Veritate.")
    print("1. Create Model")
    print("2. Use Pre-Existing Model")
    print("3. Exit")
    choice=str(input())
    if choice=="1":
        print("Initializing Classifier...")
        C=str(input("What regularization degree? "))
        if C[0].isnumeric() and C[1]=="." and C[2:].isnumeric():
            C=float(C)
        else:
            C=float(1.0)
            print("Invalid value entered, default value of 1.0 set.")
        fit_intercept=str(input("Fit an intercept? "))
        if fit_intercept[0]=="y":
            fit_intercept=True
        elif fit_intercept[0]=="n":
            fit_intercept=False
        else:
            fit_intercept=False
            print("Invalid value entered, default of no intercept applied.")
        max_iter=str(input("Maximum number of iterations? "))
        if max_iter.isnumeric():
            max_iter=int(max_iter)
        else:
            max_iter=1000
            print("Invalid value entered, default value of 1000 set.")
        validation_fraction=str(input("Fraction of data to use as validation? "))
        if validation_fraction[0].isnumeric() and validation_fraction[1]=="." and validation_fraction[2:].isnumeric():
            if 0<float(validation_fraction)<1:
                validation_fraction=float(validation(fraction))
            else:
                validation_fraction=0.1
                print("Invalid value entered, default value of 0.1 set.")
        else:
            validation_fraction=0.1
            print("Invalid value entered, default value of 0.1 set.")
        shuffle=str(input("Shuffle training data each time? "))
        if shuffle[0]=="y":
            shuffle=True
        elif shuffle[0]=="n":
            shuffle=False
        else:
            shuffle=True
            print("Invalid value entered, default of shuffling applied.")
        if shuffle:
            random_state=str(input("What seed state to set when shuffling data? "))
            if random_state.isnumeric():
                random_state=int(random_state)
            else:
                random_state=None
                print("Invalid value entered, default value of None used.")
        warm_start=str(input("Reuse solutions from each validation iteration for future validations? "))
        if warm_start[0]=="y":
            warm_start=True
        elif warm_start[0]=="n":
            warm_start=False
        else:
            warm_start=False
            print("Invalid value entered, default set to not use previous solutions.")
        primary_directory=str(input("Enter path of working directory to use: "))
        try:
            os.chdir(primary_directory)
            print("New working directory set.")
        except:
            print("Specified directory not found, current working directory will be used.")
        data=str(input("Enter the exact name of the .csv file to be used: "))
        try:
            df=pd.read_csv(data+".csv")
            test_size=str(input("What fraction of data to be used as the test set? "))
            if test_size[0].isnumeric() and test_size[1]=="." and test_size[2:].isnumeric():
                if 0<float(test_size)<1:
                    test_size=float(test_size)
                else:
                    print("Invalid value entered, test_size set to 0.2.")
            else:
                print("Invalid value entered, test_size set to 0.2.")
            response=str(input("What column to use as response? "))
            columns=df.columns
            if response in columns:
                labels=df[response]
                main_text=str(input("What column to use as main predictor? "))
                if main_text in columns:
                    x_train, x_test, y_train, y_test=train_test_split(df[main_text], labels, test_size=test_size, random_state=7)
                    x_train.sort_values()
                    x_test.sort_values()
                    y_train.sort_values()
                    y_test.sort_values()                    
                    tfidf_vectorizer=TfidfVectorizer(stop_words="english", max_df=0.7)
                    tfidf_train=tfidf_vectorizer.fit_transform(x_train)
                    tfidf_test=tfidf_vectorizer.transform(x_test)
                    pac=PassiveAggressiveClassifier(C=C, fit_intercept=fit_intercept, max_iter=max_iter, validation_fraction=validation_fraction, shuffle=shuffle, random_state=random_state, warm_start=warm_start)
                    pac.fit(tfidf_train, y_train)
                    y_pred=pac.predict(tfidf_test)
                    score=accuracy_score(y_test, y_pred)
                    print("Accuracy of the created model is "+ str(round(score*100, 2))+"%.")
                    save=str(input("Save model? "))
                    if save[0]=="y":
                        directory=str(input("Enter path of directory to save to: "))
                        try:
                            os.chdir(directory)
                            name=str(input("Enter the name of the model: "))
                            try:
                                f=open(name+".txt")
                                f.close()
                                overwrite=str(input("Model with the same name already exists! Overwrite existing model? "))
                                if overwrite[0]=="y":
                                    os.remove(name+".txt")
                                    handle=open(name+".txt", "w")
                                    handle.write(str(name)+"\n")
                                    handle.write(str(C)[0]+"\n")
                                    handle.write(str(fit_intercept)[0]+"\n")
                                    handle.write(str(max_iter)+"\n")
                                    handle.write(str(validation_fraction)+"\n")
                                    handle.write(str(shuffle)[0]+"\n")
                                    handle.write(str(random_state)+"\n")
                                    handle.write(str(warm_start)+"\n")
                                    handle.write(str(primary_directory)+"\n")
                                    handle.write(str(data)+".csv"+"\n")
                                    handle.write(str(test_size)+"\n")
                                    handle.write(str(response)+"\n")
                                    handle.write(str(main_text))
                                    handle.close()
                                    print("Existing model overwritten.")
                                elif overwrite[0]=="n":
                                    print("Returning to main menu.")
                                else:
                                    print("Invalid value entered. Returning to main menu.")                                
                            except:
                                handle=open(name+".txt", "w")
                                handle.write(str(name)+"\n")
                                handle.write(str(C)[0]+"\n")
                                handle.write(str(fit_intercept)[0]+"\n")
                                handle.write(str(max_iter)+"\n")
                                handle.write(str(validation_fraction)+"\n")
                                handle.write(str(shuffle)[0]+"\n")
                                handle.write(str(random_state)+"\n")
                                handle.write(str(warm_start)+"\n")
                                handle.write(str(primary_directory)+"\n")
                                handle.write(str(data)+".csv"+"\n")
                                handle.write(str(test_size)+"\n")
                                handle.write(str(response)+"\n")
                                handle.write(str(main_text))
                                handle.close()
                                print("Model saved.")
                        except:
                            print("Directory not found. Returning to main menu.")
                    elif save[0]=="n":
                        print("Returning to main menu.")
                    else:
                        print("Invalid choice entered. Returning to main menu.")
                else:
                    print("No such predictor found. Returning to main menu.")
            else:
                print("No such response found. Returning to main menu.")
        except:
            print("File not found in the current directory!")
            print("Returning to main menu.")
    elif choice=="2":
        directory=str(input("Enter directory to choose models from: "))
        try:
            os.chdir(directory)
            model=str(input("Choose model to use. "))
            try:
                handle=open(model+".txt", "r")
                contents=handle.readlines()
                handle.close()
                for i in range(len(contents)-1):
                    contents[i]=contents[i][:-1]
                C=float(contents[1])
                if contents[2]=="T":
                    fit_intercept=True
                else:
                    fit_intercept=False
                max_iter=int(contents[3])
                validation_fraction=float(contents[4])
                if contents[5]=="T":
                    shuffle=True
                else:
                    shuffle=False
                if contents[6]=="None":
                    random_state=None
                else:
                    random_state=int(contents[6])
                if contents[7]=="T":
                    warm_start=True
                else:
                    warm_start=False
                os.chdir(contents[8])
                df=pd.read_csv(contents[9])
                test_size=float(contents[10])
                response=contents[11]
                main_text=contents[12]
                labels=df[response]
                x_train, x_test, y_train, y_test=train_test_split(df[main_text], labels, test_size=test_size, random_state=7)
                x_train.sort_values()
                x_test.sort_values()
                y_train.sort_values()
                y_test.sort_values()
                tfidf_vectorizer=TfidfVectorizer(stop_words="english", max_df=0.7)
                tfidf_train=tfidf_vectorizer.fit_transform(x_train)
                tfidf_test=tfidf_vectorizer.transform(x_test)
                pac=PassiveAggressiveClassifier(C=C, fit_intercept=fit_intercept, max_iter=max_iter, validation_fraction=validation_fraction, shuffle=shuffle, random_state=random_state, warm_start=warm_start)
                pac.fit(tfidf_train, y_train)
                print("Model successfully recovered.")
                directory=str(input("Enter path of the directory where the news item is: "))            
                try:
                    os.chdir(directory)
                    news_type=str(input("What type of news item to validate? (1 for Article, 2 for Report, 0 for Neither) "))
                    if news_type=="0":
                        name=str(input("Enter name of news item to retrieve: "))
                        try:
                            handle=open(name+".txt", "r", encoding="utf8")
                            news=handle.readlines()
                            handle.close()
                            for i in range(len(news)-1):
                                if news[i]!="?":
                                    news[i]=news[i][:-1]
                            item=NewsItem(title=news[0], date=news[1], org=news[2], author=news[3], valid=bool(news[4]), content=" ".join(news[5:]))
                            print("Predictor for this model is '" + main_text + "'.")
                            x=str(input("What feature of news item to use as predictor? "))
                            attributes={}
                            for attribute in inspect.getmembers(item):
                                if not attribute[0].startswith("_"):
                                    if not inspect.ismethod(attribute[1]):
                                        attributes[attribute[0]]=attribute[1]
                            if x in list(attributes.keys()):
                                single=pd.Series([attributes[x]], index=[1])
                                test=tfidf_vectorizer.transform(single)
                                result=pac.predict(test)
                                if result[0]=="REAL":
                                    if item.valid:
                                        print("The model predicted '"+str(item.title)+"' to be true, and it was correct.")
                                    else:
                                        print("The model predicted '"+str(item.title)+"' to be true, and it was incorrect.")
                                else:
                                    if item.valid:
                                        print("The model predicted '"+str(item.title)+"' to be false, and it was incorrect.")
                                    else:
                                        print("The model predicted '"+str(item.title)+"' to be false, and it was correct.")
                            else:
                                print("Chosen predictor not an attribute of the news item. Returning to main menu.")         
                        except:
                            print("News item not found! Returning to main menu.")
                    elif news_type=="1":
                        name=str(input("Enter name of article to retrieve: "))
                        try:
                            handle=open(name+".txt", "r", encoding="utf8")
                            news=handle.readlines()
                            handle.close()
                            for i in range(len(news)-1):
                                if news[i]!="?":
                                    news[i]=news[i][:-1]
                            item=Article(title=news[0], date=news[1], org=news[2], author=news[3], valid=bool(news[4]), bias=bool(news[5]), content=" ".join(news[6:]))
                            print("Predictor for this model is '" + main_text + "'.")
                            x=str(input("What feature of the article to use as predictor? "))
                            attributes={}
                            for attribute in inspect.getmembers(item):
                                if not attribute[0].startswith("_"):
                                    if not inspect.ismethod(attribute[1]):
                                        attributes[attribute[0]]=attribute[1]
                            if x in list(attributes.keys()):
                                single=pd.Series([attributes[x]], index=[1])
                                test=tfidf_vectorizer.transform(single)
                                result=pac.predict(test)
                                if result[0]=="REAL":
                                    if item.valid:
                                        if item.bias:
                                            print("The model predicted the article '"+str(item.title)+"' to be true despite its bias, and it was correct.")
                                        else:
                                            print("The model predicted the article '"+str(item.title)+"' to be true, partially due to its lack of bias, and it was correct.")
                                    else:
                                        if item.bias:
                                            print("The model predicted the article '"+str(item.title)+"' to be true despite its bias, and it was incorrect.")
                                        else:
                                            print("The model predicted the article '"+str(item.title)+"' to be true, partially due to its lack of bias, and it was incorrect.")
                                else:
                                    if item.valid:
                                        if item.bias:
                                            print("The model predicted the article '"+str(item.title)+"' to be false, partially due to its bias, and it was correct.")
                                        else:
                                            print("The model predicted the article '"+str(item.title)+"' to be false despite its lack of bias, and it was correct.")
                                    else:
                                        if item.bias:
                                            print("The model predicted the article '"+str(item.title)+"' to be false, partially due to its bias, and it was incorrect.")
                                        else:
                                            print("The model predicted the article '"+str(item.title)+"' to be false despite its lack of bias, and it was incorrect.")
                            else:
                                print("Chosen predictor not an attribute of the news item. Returning to main menu.")
                        except:
                            print("Article not found! Returning to main menu.")                        
                    elif news_type=="2":
                        name=str(input("Enter name of report to retrieve: "))
                        try:
                            handle=open(name+".txt", "r", encoding="utf8")
                            news=handle.readlines()
                            handle.close()
                            for i in range(len(news)-1):
                                if news[i]!="?":
                                    news[i]=news[i][:-1]
                            item=NewsItem(title=news[0], date=news[1], org=news[2], author=news[3], valid=bool(news[4]), accuracy=float(news[5]), content=" ".join(news[6:]))
                            print("Predictor for this model is '" + main_text + "'.")
                            x=str(input("What feature of the report to use as predictor? "))
                            attributes={}
                            for attribute in inspect.getmembers(item):
                                if not attribute[0].startswith("_"):
                                    if not inspect.ismethod(attribute[1]):
                                        attributes[attribute[0]]=attribute[1]
                            if x in list(attributes.keys()):
                                single=pd.Series([attributes[x]], index=[1])
                                test=tfidf_vectorizer.transform(single)
                                result=pac.predict(test)
                                print("This report is " + str(100*item.accuracy) + "% accurate.")
                                if result[0]=="REAL":
                                    if item.valid:
                                        print("The model predicted the report '"+str(item.title)+"' to be true, and it was correct.")
                                    else:
                                        print("The model predicted the report '"+str(item.title)+"' to be true, and it was incorrect.")
                                else:
                                    if item.valid:
                                        print("The model predicted the report'"+str(item.title)+"' to be false, and it was incorrect.")
                                    else:
                                        print("The model predicted the report'"+str(item.title)+"' to be false, and it was correct.")
                            else:
                                print("Chosen predictor not an attribute of the news item. Returning to main menu.")
                        except:
                            print("Report not found! Returning to main menu.")                        
                    else:
                        print("Invalid choice entered. Returning to main menu.")                        
                except:
                    print("Directory not found! Returning to main menu.")
            except:
                print("Model not found, returning to main menu.")
        except:
            print("Directory not found! Returning to main menu.")
    elif choice=="3":
        print("Thank you for using Salva Veritate. Until next time.")
        break
    else:
        print("Please enter a valid choice.")