from flask import Flask, render_template, request, send_file, redirect, url_for
import math
import scipy.stats as stats 
from werkzeug.utils import secure_filename
import pandas as pd 
import random 
from sklearn.cluster import KMeans 
from sklearn.cluster import Birch
from kneed import KneeLocator
import os 
from pathlib import Path


sample_size_glob = 0
file_name = None 
Flask_App = Flask(__name__) # Creating our Flask Instance

#To display the intial page 
@Flask_App.route('/', methods=['GET'])
def index():
    return render_template('index1.html')

#To display the calculations of Evan Miller's Sample Size 
@Flask_App.route('/operation_result/', methods=['POST'])
def operation_result():
    error = None
    result = None

    #To take the input of Baseline Conversion rate 
    first_input = request.form['Input1']  
    #To take the input of Minimum detectable rate
    second_input = request.form['Input2']
    #To take the input of statistical power 
    third_input = request.form['Input3']
    #To take the input of statistical level 
    fourth_input = request.form['Input4']

    input1 = float(first_input)
    input2 = float(second_input)
    input3 = float(third_input)
    input4 = float(fourth_input)

    if (input1 > 49):
        input1 = 100-input1
    p1 = input1/100
    p2 = (input2+input1)/100
    alpha = (input4/100)/2 
    beta = 1- (input3/100)
    z_score_alpha = stats.norm.ppf(1-alpha)
    z_score_beta = stats.norm.ppf(1-beta)
    part_1 = z_score_alpha*math.sqrt((2*p1*(1-p1)))
    part_2 = z_score_beta*math.sqrt(p1*(1-p1) + p2*(1-p2))
    deno = math.pow(p2 - p1, 2) 
    n = (math.pow((part_1+part_2),2))/(deno)
    sample_size_glob = n
    #result = round(n)
    #return redirect(url_for('sampling_result', name = result))
    return render_template(
           'index1.html',
            input1=input1,
            input2=input2,
            result= sample_size_glob,
            calculation_success=True
        )

@Flask_App.route('/download')
def download():
    THIS_FOLDER = Path(__file__).parent.resolve()
    #path = 'C://Users//Sandhya//OneDrive//Desktop//AB Testing_Folder//ABTesting_v2//ABTesting_V2.0//AB_testing_V3'
    #default = '//Blood Transfusion Service Centre Dataset.csv'
    #To download the available file in the folder only 
    if (os.path.exists(str(THIS_FOLDER)+"\\result_SystematicSample.csv")): 
        second = "\\result_SystematicSample.csv"
    elif (os.path.exists(str(THIS_FOLDER)+"\\result_StratifiedSample_KMeans.csv")): 
        second = "\\result_StratifiedSample_KMeans.csv"
    elif (os.path.exists(str(THIS_FOLDER)+"\\result_StratifiedSample_BIRCH.csv")) : 
        second = "\\result_StratifiedSample_BIRCH.csv" 
    else : 
        second = "\\result_RandomSample.csv"
    final_path = str(THIS_FOLDER) + str(second)
    return send_file(final_path, as_attachment=True)

#To take the input in a dropdown for sampling - Simple Random sampling, Stratified sampling - KMeans, Stratified Sampling - BIRCH, Systematic Sampling
@Flask_App.route('/sampling_result/', methods=['POST','GET'])
def sampling_result():
    error = None
    result = None
    first_input = request.form['operator']  
    input1 = first_input
    sample_size = 100
    THIS_FOLDER = Path(__file__).parent.resolve()
    final_data = str(THIS_FOLDER)+"\\Blood Transfusion Service Centre Dataset.csv"
    data = pd.read_csv(final_data)

    if (input1 == "SR"):
        #data = pd.read_csv(r"C:\Users\Sandhya\OneDrive\Desktop\AB Testing_Folder\ABTesting_v2\ABTesting_V2.0\AB_testing_V3\Blood Transfusion Service Centre Dataset.csv")
        #sample_size = 100
        def Random_sample(data,sample_size): 
           new_data_random = pd.DataFrame()
           #sample_size = sample_size 
           file = ""
           for i in range(sample_size): 
                 number = random.randint(1,data.shape[0])
                 new_data_random = pd.concat([new_data_random,data.iloc[number:number+1]])
           #path = "C:\\Users\\Sandhya\\OneDrive\\Desktop\\AB Testing_Folder\\ABTesting_v2\\ABTesting_V2.0\\AB_testing_V3"
           file = new_data_random.to_csv(str(THIS_FOLDER)+"\\result_RandomSample.csv") 
           if (os.path.exists(str(THIS_FOLDER)+"\\result_SystematicSample.csv")): 
                os.remove(str(THIS_FOLDER)+"\\result_SystematicSample.csv")
           if (os.path.exists(str(THIS_FOLDER)+"\\result_StratifiedSample_BIRCH.csv")): 
               os.remove(str(THIS_FOLDER)+"\\result_StratifiedSample_BIRCH.csv")      
           if (os.path.exists(str(THIS_FOLDER)+"\\result_StratifiedSample_KMeans.csv")): 
               os.remove(str(THIS_FOLDER)+"\\result_StratifiedSample_KMeans.csv")      
           return file 
        Random_sample(data,sample_size)
        result = sample_size

    elif (input1 == "SyC"):
        #data = pd.read_csv(r"C:\Users\Sandhya\OneDrive\Desktop\AB Testing_Folder\ABTesting_v2\ABTesting_V2.0\AB_testing_V3\Blood Transfusion Service Centre Dataset.csv")
        #sample_size = 100 
        def Systematic_sample(data,sample_size): 
             #sample_size = sample_size
             k = int (data.shape[0]/sample_size)
             new_data_systematic = pd.DataFrame()
             for i in range(1, data.shape[0],k):
                new_data_systematic = pd.concat([new_data_systematic,data.iloc[i:i+1]])
                if(new_data_systematic.shape[0]==sample_size): 
                 break 
             #path = "C:\\Users\\Sandhya\\OneDrive\\Desktop\\AB Testing_Folder\\ABTesting_v2\\ABTesting_V2.0\\AB_testing_V3"
             file = new_data_systematic.to_csv(str(THIS_FOLDER)+"\\result_SystematicSample.csv") 
             if (os.path.exists(str(THIS_FOLDER)+"\\result_StratifiedSample_BIRCH.csv")): 
                 os.remove(str(THIS_FOLDER)+"\\result_StratifiedSample_BIRCH.csv")
             if (os.path.exists(str(THIS_FOLDER)+"\\result_StratifiedSample_KMeans.csv")): 
                 os.remove(str(THIS_FOLDER)+"\\result_StratifiedSample_KMeans.csv")                 
             if (os.path.exists(str(THIS_FOLDER)+"\\result_RandomSample.csv")): 
                 os.remove(str(THIS_FOLDER)+"\\result_RandomSample.csv")                
             return file 
             #return new_data_systematic
        Systematic_sample(data,sample_size)
        result = sample_size

    elif (input1 == "StCK") :
        #data = pd.read_csv(r"C:\Users\Sandhya\OneDrive\Desktop\AB Testing_Folder\ABTesting_v2\ABTesting_V2.0\AB_testing_V3\Blood Transfusion Service Centre Dataset.csv")
        #sample_size = 100 
        def Stratified_sample_KMeans(data,sample_size):
             new_data_stratified = data
             sse = []
             #To check the elbow of Kmeans 
             for i in range(2,11):
                  kmeans = KMeans(n_clusters=i)
                  kmeans.fit(new_data_stratified)
                  sse.append(kmeans.inertia_)
             kl = KneeLocator(range(2, 11), sse, curve="convex", direction="decreasing")
             #To create the final clusters 
             kmeans = KMeans(n_clusters=kl.elbow)
             kmeans.fit(new_data_stratified)
             new_data_stratified["label"] = list(kmeans.labels_)
             l_num = {}
             for i in range(0,kl.elbow): 
               l_num[i] = (round(((len(new_data_stratified[new_data_stratified.label==i])/new_data_stratified.shape[0])*sample_size)))
              # print(kl.elbow)
             l_data = {}
             for i in  range(0,kl.elbow): 
               l_data[i] = new_data_stratified[new_data_stratified.label==i].groupby('label', group_keys=False).apply(lambda x: x.sample(l_num[i]))
             final_data = []
             final_data = pd.concat([l_data[0],l_data[1],l_data[2],l_data[3]])
             #Random_sample(data,sample_size).to_csv('C:/Users/Sandhya/Downloads/result.csv')
             #path = "C:\\Users\\Sandhya\\OneDrive\\Desktop\\AB Testing_Folder\\ABTesting_v2\\ABTesting_V2.0\\AB_testing_V3"
             file = final_data.to_csv(str(THIS_FOLDER)+"\\result_StratifiedSample_KMeans.csv")
             if (os.path.exists(str(THIS_FOLDER)+"\\result_SystematicSample.csv")): 
                 os.remove(str(THIS_FOLDER)+"\\result_SystematicSample.csv")
             if (os.path.exists(str(THIS_FOLDER)+"\\result_RandomSample.csv")): 
                 os.remove(str(THIS_FOLDER)+"\\result_RandomSample.csv")       
             if (os.path.exists(str(THIS_FOLDER)+"\\result_StratifiedSample_BIRCH.csv")): 
                 os.remove(str(THIS_FOLDER)+"\\result_StratifiedSample_BIRCH.csv")                             
             return file 
             #return final_data
        Stratified_sample_KMeans(data,sample_size)
        result = sample_size

    else : 
        #data = pd.read_csv(r"C:\Users\Sandhya\OneDrive\Desktop\AB Testing_Folder\ABTesting_v2\ABTesting_V2.0\AB_testing_V3\Blood Transfusion Service Centre Dataset.csv")
        #sample_size = 100 
        def Stratified_sample_Birch(data, sample_size):
            model = Birch()
            model.fit(data)
            labels = model.predict(data)
            #To add the column labels to the dataframe available 
            data['labels']= labels 
            #To find the total number of clusters in the dataset from the column - labels 
            total = len(data.labels.value_counts())
            #To find how many number of clusters are required in the sample size based on the percentage of the actual dataset in the vairble l_num
            l_num = {}
            for i in range(0,total-1): 
                l_num[i] = (round(((len(data[data.labels==i])/data.shape[0])*sample_size)))
            #To find the 
            l_data = {}
            #To declare the variable new_array for the record of the clusters who's value in the sample data is not zero
            new_array = []
            for i in  range(0,total-1): 
                if(l_num[i]!=0):
                 #To sample data which is not zero using the if function 
                    l_data[i] = data[data.labels==i].groupby('labels').apply(lambda x: x.sample(l_num[i]))
                    new_array.append(i)
                 #To concat all the variable into the final data for the final sample dataset 
            final_data = []
            final_data = pd.concat([l_data[i] for i in new_array], ignore_index = True)
            #return final_data
            #path = "C:\\Users\\Sandhya\\OneDrive\\Desktop\\AB Testing_Folder\\ABTesting_v2\\ABTesting_V2.0\\AB_testing_V3"
            file = final_data.to_csv(str(THIS_FOLDER)+"\\result_StratifiedSample_BIRCH.csv")
            if (os.path.exists(str(THIS_FOLDER)+"\\result_SystematicSample.csv")): 
               os.remove(str(THIS_FOLDER)+"\\result_SystematicSample.csv")
            if (os.path.exists(str(THIS_FOLDER)+"\\result_RandomSample.csv")): 
               os.remove(str(THIS_FOLDER)+"\\result_RandomSample.csv")   
            if(os.path.exists(str(THIS_FOLDER)+"\\result_StratifiedSample_KMeans.csv")):   
               os.remove(str(THIS_FOLDER)+"\\result_StratifiedSample_KMeans.csv")                    
            return file 
        Stratified_sample_Birch(data,sample_size)
        result = sample_size
        
    return render_template(
           'index1.html',
            input1=input1,
            sample_result=result,
            sample_success=True
        )

@Flask_App.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        #sample_size = 100
        f = request.files['file']
        f.save(f.filename)  
        #data = pd.read_csv(f)
        #print(sample_size)
        return render_template("index1.html", name = f.filename)  

#@Flask_App.route('/select', methods=['POST', 'GET'])
#def select():
 #   value = request.form.get('operator')  

if __name__ == '__main__':
    Flask_App.debug = True
    Flask_App.run(host='0.0.0.0', port=5010)
