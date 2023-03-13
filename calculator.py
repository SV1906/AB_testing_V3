from flask import Flask, render_template, request, send_file, redirect, url_for, abort, flash, session
#from flask_session import Session
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
import numpy as np
from csv import writer



file_name = None 
Flask_App = Flask(__name__) # Creating our Flask Instance
Flask_App.secret_key = 'random string'
Flask_App.config['SESSION_PERMANENT'] = False
Flask_App.config["SESSION_TYPE"] = "filesystem"
#Session(Flask_App)


#To display the intial page 
@Flask_App.route('/', methods=['GET'])
def index():
    return render_template('Index1.html')

@Flask_App.route('/Index')
def New_Testing():
    return render_template('New_Testing.html')

#
@Flask_App.route('/Suggesting_method', methods = ['POST'])
def suggesting_method():
        #Preprocessing Technique
        THIS_FOLDER = Path(__file__).parent.resolve()
        final_data = str(THIS_FOLDER)+"\\Blood Transfusion Service Centre Dataset.csv"
        original_data = pd.read_csv(final_data)
        data_dummy = original_data.copy()
        data_dummy = data_dummy.drop_duplicates()
        missing_values_count = data_dummy.isnull().sum()
        names = np.where(missing_values_count > 0)
        percent_missing = (missing_values_count.sum()/np.product(data_dummy.shape))*100       
        if(percent_missing < 5): 
            data_dummy = data_dummy.dropna()
        else: 
            for i in data_dummy:
                if (data_dummy[i].dtypes == "int64" or data_dummy[i].dtypes == "float64"):
                    if(data_dummy[i].skew()>=0.5 or data_dummy[i].skew()<=-0.5):
                        #Median; Skewed;range : more than 0.5 and less than -0.5
                        mean_value = data_dummy[i].mean()
                        data_dummy[i].fillna(value = mean_value, inplace = True)
                    else : 
                         #Mean; Not Skewed: Symmetric; range : more than -0.5 and less than 0.5
                         median_value = data_dummy[i].median()
                         data_dummy[i].fillna(value = median_value, inplace = True)
        listing_int = []
        listing_string_object = []
        j = 0
        for i in data_dummy.dtypes: 
            if (i == 'int64' or i == 'float64'):
                listing_int.append(data_dummy.columns[j])  
            else: 
                listing_string_object.append(data_dummy.columns[j])
            j = j+1
        Q1 = data_dummy.quantile(0.25)
        Q3 = data_dummy.quantile(0.75)
        IQR = Q3 - Q1
        data_dummy = data_dummy[~((data_dummy< (Q1 - 1.5 * IQR)) |(data_dummy > (Q3 + 1.5 * IQR))).any(axis=1)]
        for k in listing_int:
            unique_values = len(data_dummy[k].unique())
            if (unique_values == len(data_dummy)):
                data_dummy = data_dummy.drop([k], axis=1)
                listing_int.remove(k)
                continue 
            elif (unique_values<=5) : 
                data_dummy = pd.get_dummies(data_dummy, columns = [k]) 
                listing_int.remove(k)
                continue  
            if(((data_dummy[k].skew()) > 0.5) or ((data_dummy[k].skew()) < -0.5)): 
                data_dummy[k] = np.log(data_dummy[k])
        for k in listing_string_object: 
            data_dummy[k] = data_dummy[k].apply(str.lower)
            data_dummy[k] = data_dummy[k].apply(str.strip)
            unique_values = len(data_dummy[k].unique())
            if (unique_values<=5): 
                data_dummy = pd.get_dummies(data_dummy, columns = [k])  
            else : 
                data_dummy = data_dummy.drop([k], axis=1)
        Sample = "" 
        if (len(original_data)-len(data_dummy) < 0.20*len(original_data)):
            if(len(original_data.columns) > 1):   
              #  after_strat = input("Do you want to choose a sampling method other than Stratified sampling? ").lower()
              #  if (after_strat == "yes" or after_strat == "y"):
                    #print("Random Sampling or Systematic Sampling")
               #     random_systematic = input("Do you want to use Random Sampling? ").lower()
                #    if(random_systematic == "yes" or random_systematic == "y"):
                        #print("Random Sampling")
                        Sample = "Random Sampling"
                 #   else: 
                        #print("Systematic Sampling")
                  #      Sample = "Systematic Sampling"
              #  else: 
                    #print("Stratified Sampling") 
               #     if (np.product(data_dummy.shape) > 50000):
                        #print("stratified Sampling - BIRCH")
                #        Sample = "Stratified Sampling - BIRCH"
                 #   else : 
                        #print("Stratified Sampling - Kmeans")
                  #      Sample = "Stratified Sampling - Kmeans"
            else: 
                #print("Random Sampling or Systematic Sampling")
             #   random_systematic = input("Do you want to use Random Sampling? ").lower()
             #   if(random_systematic == "yes" or random_systematic == "y"):
                    #print("Random Sampling")
             #       Sample = "Random Sampling"
             #   else: 
                    #print("Systematic Sampling")
                    Sample = "Systematic Sampling"
        else : 
        #    cluster = input("Is the data uploaded in clusters? ").lower()
        #    if(cluster == "yes" or cluster == "y"): 
                #print("Cluster Sampling")
         #       Sample = "Cluster Sampling"
         #   else : 
                #print("Systematic Sampling")
                Sample = "Systematic Sampling"
        return render_template(
           'New_Testing.html',
            section = 'section_sampling',
            Sample_Suggested = "Sampling technique : " + Sample
        )


#To display the calculations of Evan Miller's Sample Size 
@Flask_App.route('/operation_result_basic/', methods=['POST'])
def operation_result_basic():

    error = None
    result = None
    #To take the input of Confidence Interval 
    first_input = request.form['Input1']  
    #To take the input of Margin of Error
    second_input = request.form['Input2']

    CI = float(first_input)/100
    MOE = float(second_input)/100
    z_score_CI = round(stats.norm.ppf(1-(1-CI)/2),2) 
    result = round(0.25/math.pow((MOE/z_score_CI),2))

    result_array = [result]
    THIS_FOLDER = Path(__file__).parent.resolve()
    file = (str(THIS_FOLDER)+"\\Variables.csv")
    with open(file, 'w') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(result_array)
        f_object.close()
        
    return render_template(
       'New_Testing.html',
            basic_sampling_result= result,
            section = 'section_samplesize'
            
        )

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
    result = round(n)
    result_array = [result]
    THIS_FOLDER = Path(__file__).parent.resolve()
    file = (str(THIS_FOLDER)+"\\Variables.csv")
    with open(file, 'w') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(result_array)
        f_object.close()

    return render_template(
       'New_Testing.html',
            sampling_result= result,
            section = 'section_samplesize',
            calculation_success=True, 
           
        )

@Flask_App.route('/Hypothesis/', methods=['POST'])
def Hypothesis():
    error = None
    result = None
    #To take the input of Baseline Conversion rate 
    first_input = request.form['Insight_input']  
    #To take the input of Minimum detectable rate
    second_input = request.form['Campaign_input']
    #To take the input of statistical power 
    third_input = request.form['Result_input']
    #To take the input of statistical level 


    input1 = first_input
    input2 = second_input
    input3 = third_input

    output = "Given that "+ input1 +", changing " + input2 + " will result in "+ input3
    return render_template(
       'New_Testing.html',
            hypothesis_result= output,
           ## section = 'section_samplesize',
           ## calculation_success=True, 
           ## scroll = 'something'
        )

@Flask_App.route('/User_Input/', methods=['POST'])
def User_Input():
    
    error = None
    result = None


    #To take the input of Baseline Conversion rate 
   # first_input = request.form['Input1']  
    #To take the input of Minimum detectable rate
   # second_input = request.form['Input2']
    #To take the input of statistical power 
   # third_input = request.form['Campaign_type_input']
    #To take the input of statistical level 
  #  fourth_input = request.form['Input4']
  #  fifth_input = request.form['Input5']
  #  sixth_input = request.form['Input6']
   # seventh_input = request.form['Input7']
   # eighth_input = request.form['Conversion_metric_input']
  #  nineth_input = request.form['Input9']

    Campaign_name = request.form['campaign_name_input']

    Campaign_start = request.form['campaign_start_date']
    Campaign_end = request.form['campaign_end_date']
    Conversion_metric = request.form['Conversion_metric_input']
    Campaign_type = request.form['Campaign_type_input']  
    Date_of_requirement = request.form['Date_of_requirement']


    THIS_FOLDER = Path(__file__).parent.resolve()

    file = (str(THIS_FOLDER)+"\\Final_Excel_1.csv")
    df = pd.read_csv(file)
    index_len = df.shape[0]+1
    
    array_output = []

    array_output.append(index_len)
    array_output.append(Date_of_requirement)
    array_output.append(Campaign_name)
    array_output.append(Campaign_type)
    array_output.append(Campaign_start)
    array_output.append(Campaign_end)
    array_output.append(Conversion_metric)

    with open(file, 'a' , newline ='') as f_object:
          writer_object = writer(f_object)
          writer_object.writerow(array_output)
          f_object.close()

    return render_template(
           'New_Testing.html',
            array_result = array_output,
            calculation_success=True,
            open_section = "Download_section"
        )


@Flask_App.route('/Col_Input/', methods=['POST'])
def Col_Input():

    array_col_input_2 = []


    op1_checked, op2_checked, op3_checked, op4_checked = False, False, False, False
    if request.form.get("ETB") :
        op1_checked = True
        array_col_input_2.append("ETB")
    if request.form.get("NTB"):
        op2_checked = True
        array_col_input_2.append("NTB")
    if request.form.get("PTB"):
        op3_checked = True
        array_col_input_2.append("PTB")
    if request.form.get("EMI CARDED"):
        op4_checked = True
        array_col_input_2.append("EMI CARDED")

    array_col_input =[]
    array_col_input.append(op1_checked)
    array_col_input.append(op2_checked)
    array_col_input.append(op3_checked)
    array_col_input.append(op4_checked)

    THIS_FOLDER = Path(__file__).parent.resolve()
    final_data = str(THIS_FOLDER)+"\\Blood Transfusion Service Centre Dataset.csv"
    sample_size_data = str(THIS_FOLDER) + "\\Variables.csv"
    sample_size_data = pd.read_csv(sample_size_data)
    sample_size = int(sample_size_data.columns[0])
    data = pd.read_csv(final_data)

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



    
   # def Stratified_Sampling(sample_size_input, sample_size_suggested, col): 
   # if(sample_size_input < sample_size_suggested):
        #Case 1
    #    return ()
     #   print("Error : The sample size entered doesn't meet the minimum sample size requirement")
   # else : 
    #    try:
     #       #Case 2 
      #      train, test = train_test_split(df, test_size=sample_size_input, stratify=df[col].copy()) 
       #     print("Stratification is successful")
        #    print(len(test))
         #   print(len(train))
     #   except : 
            #Case 3 
      #      print("Error : Stratification is not possible on these features")  


    return render_template(
           'New_Testing.html',
           array_col_result = array_col_input,
           array_col_input_2 = array_col_input_2,
           open_section = "Stratified",
           open_section_2 = "Result_stratified"
        )
    #if (input1 > 49):
     #   input1 = 100-input1
    #p1 = input1/100
    #p2 = (input2+input1)/100
    #alpha = (input4/100)/2 
    #beta = 1- (input3/100)
    #z_score_alpha = stats.norm.ppf(1-alpha)
   # z_score_beta = stats.norm.ppf(1-beta)
    #part_1 = z_score_alpha*math.sqrt((2*p1*(1-p1)))
    #part_2 = z_score_beta*math.sqrt(p1*(1-p1) + p2*(1-p2))
    #deno = math.pow(p2 - p1, 2) 
    #n = (math.pow((part_1+part_2),2))/(deno)
    #result = round(n)
    #return render_template(
     #      'index1.html',
      #      sampling_result= result,
       #     calculation_success=True
     #   )


@Flask_App.route('/Cond_Input/', methods=['POST'])
def Cond_Input():
    array_cond_input_2 = []
    condition1_checked, condition2_checked, condition3_checked, condition4_checked, condition5_checked, condition6_checked = False, False, False, False, False, False

    if request.form.get("App Live"): 
        condition1_checked = True 
        array_cond_input_2.append("App Live")
    if request.form.get("Signed up in last 7 days"): 
        condition2_checked = True 
        array_cond_input_2.append("Signed up in last 7 days")
    if request.form.get("30 Days Active"): 
        condition3_checked = True 
        array_cond_input_2.append("30 Days Active")
    if request.form.get("Birthday Month"): 
        condition4_checked = True 
        array_cond_input_2.append("Birthday Month")
    if request.form.get("Non DEC"): 
        condition5_checked = True 
        array_cond_input_2.append("Non DEC")
    if request.form.get("DEC"): 
        condition6_checked = True 
        array_cond_input_2.append("DEC")

    array_cond_input = []
    array_cond_input.append(condition1_checked)
    array_cond_input.append(condition2_checked)
    array_cond_input.append(condition3_checked)
    array_cond_input.append(condition4_checked)
    array_cond_input.append(condition5_checked)
    array_cond_input.append(condition6_checked)

 
    return render_template(
           'New_Testing.html',
           array_cond_result = array_cond_input, 
           array_cond_input_2 = array_cond_input_2
            #array_result = array_output,
            #calculation_success=True
        )

@Flask_App.route('/download')
def download():
    THIS_FOLDER = Path(__file__).parent.resolve()
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

@Flask_App.route('/sampling_result/', methods=['POST','GET'])
def sampling_result():
    error = None
    result = None
    first_input = request.form['operator']  
    second_input = request.form['Sample_Size_input']
    input1 = first_input
    sample_size = int(second_input)
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
           'New_Testing.html',
           # result="Sample Size : " + sampling_result,
            sample_success=True
        )

#To take the input in a dropdown for sampling - Simple Random sampling, Stratified sampling - KMeans, Stratified Sampling - BIRCH, Systematic Sampling
@Flask_App.route('/sampling_select/', methods=['POST','GET'])
def sampling_select():
    error = None
    result = None
    first_input = request.form['operator']  

    THIS_FOLDER = Path(__file__).parent.resolve()
    final_data = str(THIS_FOLDER)+"\\Blood Transfusion Service Centre Dataset.csv"
    sample_size_data = str(THIS_FOLDER) + "\\Variables.csv"
    sample_size_data = pd.read_csv(sample_size_data)
    sample_size = int(sample_size_data.columns[0])
    data = pd.read_csv(final_data)

    if (first_input == "St" or first_input == "StCK" or first_input == "StCB"):
       # read session
        return render_template(
            'New_Testing.html',
             open_section = "Stratified"
        )
    
    elif (first_input == "SR"):
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
        return render_template(
           'New_Testing.html',
           # result="Sample Size : " + sampling_result,
            sample_success=True
        )
        

    else:
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
        return render_template(
           'New_Testing.html',
           # result="Sample Size : " + sampling_result,
            sample_success=True
        )
    
@Flask_App.route('/uploadfile', methods = ['POST'])  
def uploadfile():  
    if request.method == 'POST':  
        f = request.files['file']
        filename = secure_filename(f.filename)
        if filename != '':
            file_ext = os.path.splitext(filename)[1]
        if file_ext not in Flask_App.config['UPLOAD_EXTENSIONS']:
            error = 'Please upload a file with the extenstion .csv or .xlsx only. Please try again.'
            #return render_template("index1.html", error = error)
            #flash("Uploaded the wrong extension. Please upload either .csv or .xlsx") 
        else : 
            
            error = "Successfully uploaded. Please continue. "
            f.filename = "Blood Transfusion2.csv"
            f.save(f.filename)
            flash('You were successfully logged in')
    return render_template("New_Testing.html", section = 'section_upload', name = f.filename, error = error)
            
            #flash("The file has been successfully saved!")
        #return redirect(url_for('operation_result'))
    # Important Commetn 
        
       # return render_template("index1.html", section = 'about', name = f.filename, error = error)
        #return redirect(url_for('suggesting_method'))
       # return render_template(url_for('success'), name = f.filename, error = error)
        #f.save(f.filename) 
        #data = pd.read_csv(f)
        #print(sample_size)
        #return render_template("index1.html", name = f.filename, error = error)  

#@Flask_App.route('/select', methods=['POST', 'GET'])
#def select():
 #   value = request.form.get('operator') 

if __name__ == '__main__':
    Flask_App.debug = True
    Flask_App.run(host='0.0.0.0', port=5112)