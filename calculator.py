from flask import Flask, render_template, request, send_file, redirect, url_for, abort, flash, session
from datetime import date
from sklearn.model_selection import train_test_split
import math
import scipy.stats as stats 
from werkzeug.utils import secure_filename
import pandas as pd 
import random 
from kneed import KneeLocator
import os 
from pathlib import Path
import numpy as np
from csv import writer
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from datetime import datetime 
# from flask_download_btn import DownloadBtnManager, DownloadBtnMixin

file_name = None 
Flask_App = Flask(__name__) 

THIS_FOLDER = Path(__file__).parent.resolve()
final_data = str(THIS_FOLDER)+"\\DUMMY_DB.csv"
data_Original = pd.read_csv(final_data)
today = date.today()
today = today.strftime("%d-%m-%Y")
button_variable = "False" 

def features(data):
    data = data.drop(['profile_phone', 'Signup_date'],axis=1)
    new_features = []
    for feature in data.columns:
        new_feature={}
        new_feature = {'key': feature, 'value': False}
        new_features.append(new_feature)
    return new_features
 
def Stratified_On(data, remove_columns, add_columns):
    data = data.drop(['profile_phone', 'Signup_date'],axis=1)
    Stratified_on = list(data.columns.copy()) 
    res = list(set(Stratified_on) - set(remove_columns)) 
    final_stratified_on = res+ add_columns
    new_features =[]
    for feature in final_stratified_on:
        new_feature={}
        new_feature = {'key': feature, 'value': False}
        new_features.append(new_feature)
    return new_features

Features = features(data_Original)
remove_columns = ['Etb', 'Ntb','Ptb', 'Upi_Flag','Ppi_Flag','Bbps_Flag']
add_columns = ['Customer Type', 'Payments Active']
Stratification_columns = Stratified_On(data_Original, remove_columns, add_columns)

#Function to update to Master File 
def excel_update(forms, selected_features,test_size, selected_columns, today, Sub_Campaign_Names): 
    file = (str(THIS_FOLDER)+"\\Master_Sheet.csv")
    df = pd.read_csv(file, on_bad_lines='skip')
    index_len = len(df)+1
    array_output = [index_len,forms["Hypothesis"],forms['DOE'],selected_features,forms["final_result"],test_size,forms["Operator"],selected_columns,forms["CampaignName"], Sub_Campaign_Names, today, forms['CampaignStartdate'], forms['CampaignEnddate'], forms['CampaignType'], forms['ConversionMetric'], forms['ConversionPeriod']]
    with open(file, 'a' , newline ='') as f_object:
          writer_object = writer(f_object)
          writer_object.writerow(array_output)
          f_object.close()

def get_form_parameters():
    Form  = {}
    if (request.form.get("Edit_button") != None) : 
            Form["Download"] = False

         
    Form['Hypothesis'] = request.form['Hypothesis'] if (request.form.get('Hypothesis') != None) else ''
    Form['DOE'] = request.form['Doe_input'] if (request.form.get('Doe_input') != None) else 0
    Form['ConversionInterval'] = request.form['confidence_Interval_Input'] if (request.form.get('confidence_Interval_Input') != None) else 0
    Form["MarginError"] = request.form['margin_Error_Input'] if (request.form.get('margin_Error_Input') != None) else 0
    Form['Button'] = request.form['Button_id'] if (request.form.get('Button_id') != None) else ''
    Form['BaselineRate'] = request.form['baseline_Rate_Input'] if (request.form.get('baseline_Rate_Input') != None) else 0
    Form['DetectableEffect'] = request.form['detectable_Effect_Input'] if (request.form.get('detectable_Effect_Input') != None) else 0
    Form['SignificantPower'] = request.form['significance_Power_Input'] if (request.form.get('significance_Power_Input') != None) else 0
    Form['SignificantLevel'] = request.form['significance_Level_Input'] if (request.form.get('significance_Level_Input') != None) else 0
    Form["Total_cases"] = int(Form['DOE'])+1
    Form['Operator'] = 'Default'
    if (request.form.get('operator') != None) : 
        Form['Operator'] = request.form['operator']
    Form['CampaignName'] = request.form['campaign_name_input'] if (request.form.get('campaign_name_input') != None) else ''
    Form['CampaignStartdate'] = request.form['campaign_start_date'] if (request.form.get('campaign_start_date') != None) else ''
    Form['CampaignEnddate'] = request.form['campaign_end_date'] if (request.form.get('campaign_end_date') != None) else ''
    Form['CampaignType'] = request.form['Campaign_type_input'] if (request.form.get('Campaign_type_input') != None) else 'Push'
    Form['ConversionMetric'] = request.form['Conversion_metric_input'] if (request.form.get('Conversion_metric_input') != None) else 'Retention rate'
    Form['ConversionPeriod'] = request.form['Conversion_period_input'] if (request.form.get('Conversion_period_input') != None) else ''
    Form['Sum'] = 0
    # for i in range(1,int(Form["DOE"])+1):
    #     variable = "Test_case_" + str(i)
    #     if (request.form.get(variable) != None and request.form.get(variable) != '') : 
    #        Form[variable] = request.form[variable] 
    #        Form['Sum'] += int(Form[variable])
    #     else : 
    #         Form[variable] = Form["final_result"]
      

    # for i in range(1,int(Form["DOE"])+1):
    #     variable = "Campaign_Name_" + str(i)
    #     if (request.form.get(variable) != None and request.form.get(variable) != '') : 
    #        Form[variable] = request.form[variable] 
    #     else : 
    #         Form[variable] = ''

    Form['evan_millers'] = 0
    Form['final_result'] = 0
    Form['basic_result'] = 0
    Form['Records_Available'] = 0 
    Form["Sampling_Result"] =""

    Button_Section = {"Sample_Size_submit_1":"sample_size", "Sample_Size_submit_2":"sample_size","Features_button":"select_filters","Sampling_Technique_submit":"sampling_technique","Final_submit":"campaign_details","Random_button":"sampling_technique","Test_cases_button":"Test_cases","stratify_button":"sampling_technique","Button_id":"sampling_technique", "Edit_button":"hypothesis_section"}
    for key,value in Button_Section.items(): 
        if (request.form.get(key) != None) : 
            Form['location'] = value
    return Form  
    
def sample_suggest(): 
    return "Systematic Sampling"   

def verification_func(doe, sample_size, current_count):
   if ((int(doe)+1)*int(sample_size) > int(current_count)):
       return "True"
   else : 
       return "False"
   
def requiredDB(doe, sample_size):
    return (int(doe)+1)*int(sample_size)

#To Efficiently return the count of the selected features 
def db_count(data,selected): 
    try:
        return (data[selected].sum(axis=1)).value_counts()[len(selected)]
    except:
        return 0 
    
def Output_file(data_list):
    try : 
        with pd.ExcelWriter("C:\\Users\\Sandhya\\Downloads\\output.xlsx") as writer:
            for i in range(0,len(data_list)):
                data_list[i].to_excel(writer, sheet_name= "Sheet " + str(i+1))
        return True 
    except : 
        return False 

def Systematic_Sampling_result(data, test_size):
        data['Signup_date'] = pd.to_datetime(data['Signup_date'])
        data = data.sort_values("Signup_date")
        data = data.reset_index()
        data.drop(['index'], axis=1)
        threshold_control = max(test_size)*5
        Test = []
        Total = len(test_size)
        try : 
             for i in range(0, Total): 
                 step = (len(data)/test_size[i])
                 indexes = np.arange(0, len(data), step=step)
                 systematic_sample = data.iloc[indexes]
                 data.drop(systematic_sample.index, axis=0,inplace=True)
                 Test.append(systematic_sample)
             if (len(data) > threshold_control):
                 step = (len(data)/threshold_control)
                 indexes = np.arange(0, len(data), step=step)
                 systematic_sample = data.iloc[indexes]
                 Test.append(systematic_sample)
             else : 
                 Test.append(systematic_sample) 
             if (Output_file(Test)):
                 return "Sampling Successful"
             else : 
                 return "Sampling Successful but problem with saving in file."
        except : 
            return "Change the base data"

def Random_Sampling_result(data, test_size,minimum_size):
    Test = []
    Total = len(test_size)
    loop_flag = 0 
    threshold_control = max(test_size)*5 + minimum_size
    try :
        for i in range(0, Total): 
            control, test = train_test_split(data, test_size=test_size[i])
            Test.append(test)
            data = control
            loop_flag = loop_flag+1
        if (len(data) > threshold_control): 
                control, test = train_test_split(data, test_size=(threshold_control-minimum_size))
                Test.append(test)
        else : 
            Test.append(data)
        if (Output_file(Test)):
            return "Sampling Successful"
        else : 
            return "Sampling Successful but problem with saving in file."
    except : 
        return "Change the base data"
    
def Stratification_result(data, test_size,Selected, minimum_size):
    Test = []
    Total = len(test_size)
    loop_flag = 0 
    y = data[Selected]
    threshold_control = max(test_size)*5 + minimum_size
    try :
        for i in range(0, Total): 
            control, test = train_test_split(data, test_size=test_size[i], stratify= y)
            Test.append(test)
            data = control
            y = data[Selected]
            loop_flag = loop_flag+1
        if (len(data) > threshold_control): 
                control, test = train_test_split(data, test_size=(threshold_control-minimum_size), stratify= y)
                Test.append(test)
        else : 
            Test.append(data)
        if (Output_file(Test)):
            return "Sampling Successful"
        else : 
            return "Sampling Successful but problem with saving in file."
    except : 
        if (loop_flag > 0): 
            # return "Reduce the number of DOEs to atmost " + str(loop_flag) +" or change the columns to be stratified on "
            return "Reduce the number of DOEs or change the columns to be stratified on "
        else : 
            return "The least populated class in y has only 1 member. Please change the columns to be stratified on"
     
def base_data(data,Selected_DataBase): 
        for i in Selected_DataBase: 
             data = data[(data[i])==1]
        return data

def compare_size(evan_miller,basic_result): 
    return evan_miller if evan_miller >= basic_result else basic_result

def basic_Sample_Size(x,y):
    if (int(x) == 0 or int(y) == 0 ):
         return 0 
    else : 
        result = None
        CI = float(x)/100
        MOE = float(y)/100
        z_score_CI = round(stats.norm.ppf(1-(1-CI)/2),2) 
        result = round(0.25/math.pow((MOE/z_score_CI),2))
        return result

def evan_Millers(input1, input2, input3, input4):
    if (int(input1) == 0 or int(input2) == 0 or int(input3)==0 or int(input4) == 0 ):
         return 0 
    else : 
        input1 = float(input1)
        input2 = float(input2)
        input3 = float(input3)
        input4 = float(input4)
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
        return result 

#To display the intial page 
@Flask_App.route('/', methods=['GET'])
def index():
    return render_template('Index1.html')


@Flask_App.route('/Index',methods=['GET' , "POST"])
def New_Testing():

    Section_open = 1 
    verification = ""
    selected_columns = []
    test_size = []
    Sub_Campaign_Names = []
    

    if request.method == 'POST':
    

       forms = get_form_parameters()

       
       forms["Strat_result"] = ''
    #    forms["Campaign_Details_Error"] = ""

       for i in Features: 
            if request.form.get(i['key']) :
                i['value'] = True
            else : 
                i ['value'] = False 

       for i in Stratification_columns: 
            variable = "Strat" + i['key']
            if request.form.get(variable) :
                i['value'] = True
            else : 
                i ['value'] = False 
      
       if (forms["ConversionInterval"] != '' and forms["MarginError"] != ''):   
            forms ["basic_result"] = basic_Sample_Size(forms["ConversionInterval"], forms["MarginError"])
            # forms ["evan_millers"] =  evan_Millers(forms["BaselineRate"], forms["DetectableEffect"], forms["SignificantPower"], forms["SignificantLevel"])
            forms["final_result"] = compare_size(forms["evan_millers"],forms["basic_result"])
               
       if (forms["BaselineRate"] != '' and forms["DetectableEffect"] != '' and forms["SignificantPower"] != '' and forms["SignificantLevel"] != ''):           
           forms ["evan_millers"] =  evan_Millers(forms["BaselineRate"], forms["DetectableEffect"], forms["SignificantPower"], forms["SignificantLevel"])
           forms["final_result"] = compare_size(forms["evan_millers"],forms["basic_result"])

       for i in range(1,int(forms["DOE"])+1):
             variable = "Test_case_" + str(i)
             if (request.form.get(variable) != None and request.form.get(variable) != '') : 
                 forms[variable] = request.form[variable] 
                 forms['Sum'] += int(forms[variable])
             else : 
                forms[variable] = forms["final_result"]

       if (forms['DOE'] == 1): 
            Sub_Campaign_Names.append(forms['CampaignName'])
       
       for i in range(1,int(forms["DOE"])+1):
             variable = "Campaign_Name_" + str(i)
             if (request.form.get(variable) != None and request.form.get(variable) != '') : 
                    forms[variable] = request.form[variable] 
                    Sub_Campaign_Names.append(forms[variable])
             else : 
                   forms[variable] = ''
                #    Sub_Campaign_Names.append(forms[variable])
      
       selected_features = []
       for i in Features: 
             if(i['value'] == True): 
                 selected_features.append(i['key'])


    #   if (forms['DOE'] == 1): 
    #         Sub_Campaign_Names.append(forms['CampaignName'])
    #    else : 
    #         for i in range(1,int(forms["DOE"])+1):
    #             variable = "Campaign_Name_" + str(i)
    #             if (request.form.get(variable) != None and request.form.get(variable) != '') : 
    #                     Sub_Campaign_Names.append(request.form[variable])
    
       base_data_input =  base_data(data_Original, selected_features)  
       for i in range(0,int(forms["Total_cases"])-1):
                variable = "Test_case_" + str(i+1)
                test_size.insert((i+1), int(forms[variable]))

       if (forms['Operator'] != 'Default'): 
           if(forms['Operator'] == 'Random') : 
                forms["Sampling_Result"] = Random_Sampling_result(base_data_input, test_size,int(forms['final_result']))
           elif(forms['Operator'] == 'Systematic'): 
                forms["Sampling_Result"] = Systematic_Sampling_result(base_data_input, test_size)
           else : 
                for i in Stratification_columns: 
                    if (i['value'] == True):
                        dic = {'Customer Type': ['Etb', 'Ptb', 'Ntb'], 'Payments Active': ['Bbps_Flag','Upi_Flag','Ppi_Flag']}
                        if i['key'] in dic.keys():
                            selected_columns += dic[i['key']]
                        else: 
                            selected_columns.append(i['key'])      
                if (len(selected_columns) == 0):
                    forms["Error"] = "Stratification_columns"
                else : 
                    forms["Sampling_Result"] = Stratification_result(base_data_input, test_size,selected_columns, int(forms['final_result']))   

       forms["Current_Count"] = db_count(data_Original, selected_features)

       global button_variable 
       if (forms["Button"] != None or button_variable == "True"):
           button_variable = "True"
           forms["Sample_suggestion"] = sample_suggest()
        
       for i in range(1,int(forms["DOE"])+1):
            variable = "Test_case_" + str(i)
            if (request.form.get(variable) == None or request.form.get(variable) == '' or int(request.form.get(variable))< int(forms["final_result"])):
                forms[variable] = forms["final_result"]
                forms["Sum"] += forms["final_result"] 


                        
       if (forms['final_result'] != 0 and forms['Current_Count'] != ''):
            Section_open = 3
            verification = verification_func(forms["DOE"], forms["final_result"], forms["Current_Count"])
            forms["required_db"] = requiredDB(forms["DOE"],forms["final_result"])
            if (verification == "False"):
                    forms["Records_Available"] = forms['Current_Count'] - forms['final_result'] - forms['Sum']
                    if (forms["Records_Available"] < 0): 
                        forms["Records_Available"] = "Please Reduce the number of record by " + str(abs(forms["Records_Available"]))
                        forms["Error"] ="Error-lessRecords"
                    else : 
                         forms["Records_Available"] = "Records Available for Allocation : " + str(forms['Current_Count'] - forms['final_result'] - forms['Sum'])
                         Section_open = 4 

            if(verification == "True") : 
                forms['location'] = "modal_message"

       if (forms["CampaignName"] != '' and forms["CampaignStartdate"] != '' and forms["CampaignEnddate"] != '' and forms["CampaignType"] != '' and forms["ConversionMetric"] != '' and forms["ConversionPeriod"] != '' and request.form.get("Final_submit") != None):           
            forms["Campaign_Details_Error"] = ""
            if (len(Sub_Campaign_Names) != int(forms["DOE"]) and int(forms["DOE"]) != 1): 
                forms["Campaign_Details_Error"] = "Please fill in all the Sub Campaign Names"
            if (datetime.strptime(forms["CampaignStartdate"], '%Y-%m-%d') > datetime.strptime(forms["CampaignEnddate"], '%Y-%m-%d')):
               forms["Campaign_Details_Error"] = "The Campaign End Date should be after the Campaign Start DateStart date"
            if (forms["Campaign_Details_Error"] == ""):        
                try :  
                    excel_update(forms, selected_features,test_size, selected_columns, today,Sub_Campaign_Names)
                    forms ["Download"] = True  
                except : 
                    forms["ExcelUpdate"] = "There is a problem while updating the Excel. Please Try again."
       if (forms["Sampling_Result"] == "Sampling Successful"):
                Section_open = 5
       
       if (request.form.get("Download_button") != None): 
            path = "C:\\Users\\Sandhya\\Downloads\\output.xlsx"
            return send_file(path, as_attachment=True)

       return render_template('New_Testing.html', Sub_Campaign_Names = Sub_Campaign_Names, Section_open = Section_open, stratification_columns = Stratification_columns, selected_columns =selected_columns, sum = sum, Success = True, form = forms, features = Features, date = today, Verification =  verification) 
    return render_template('New_Testing.html' , Section_open = Section_open, features = Features, Stratification_columns=Stratification_columns, date = today)


# @Flask_App.route('/download')
# def download():
#     final_path = "C:\\Users\\Sandhya\\Downloads\\output.xlsx" 
#     return send_file(final_path, as_attachment=True)
  
  
if __name__ == '__main__':
    Flask_App.debug = True
    Flask_App.run(port= 5127)   
    