from flask import Flask, render_template, request, send_file
from datetime import date
from sklearn.model_selection import train_test_split
import math
import scipy.stats as stats 
import pandas as pd 
from pathlib import Path
import numpy as np
from csv import writer
from datetime import datetime 
from scipy.stats import norm
import statistics

file_name = None 
Flask_App = Flask(__name__) 

THIS_FOLDER = Path(__file__).parent.resolve()
# path = (str(THIS_FOLDER)+"\\output.xlsx")
path = Path(THIS_FOLDER) / "output.xlsx"
#To have a variable 'data_Originial' for storing the current database 
# data_Original = pd.read_csv(str(THIS_FOLDER)+"\\DUMMY_DB.csv")
data_Original = pd.read_csv(Path(THIS_FOLDER) / "DUMMY_DB.csv")


#To have a varible 'today' with today's date - required for Date of Requirement
today = date.today()
today = today.strftime("%d-%m-%Y")
#To have a varible 'button_variable' for the Suggest Sampling technique button 
button_variable = "False" 
Test_case_button = False 
#To declare a global array 'array_output_final', Sub_Campaign_Names, selected_features and selected_columns for storing the values required to update the master sheet  
array_output_final = []
Sub_Campaign_Names = []
selected_features = []
selected_columns = []

#To globally declare the variable form 
forms = {'Total_cases': 2, 'Experiment': '1', 'ConversionInterval':'' , 'MarginError':'' , 'BaselineRate':'' , 'DetectableEffect':'' , 'SignificantPower':'' , 'SignificantLevel':'' ,'Campaign_Name_1' : '', 'Campaign_Name_2': '', 'Operator' : 'Default', 'Hypothesis':'', 'CampaignName' : '','CampaignStartdate' : '','CampaignEnddate' : '', 'CampaignType':'Push' , 'ConversionMetric':'Retention Rate' ,'ConversionPeriod' : ''}

#To dynamically return the features from the database 
#Removal of the columns - Profile phone and SignUp date 
def features(data, selected_features):
    data = data.drop(['profile_phone', 'Signup_date'],axis=1)
    new_features = []
    for feature in data.columns:
        new_feature={}
        if (feature in selected_features) : 
             new_feature = {'key': feature, 'value': True}
        else : 
             new_feature = {'key': feature, 'value': False}
        new_features.append(new_feature)
    return new_features
 
#To dynamically return the columns from the database  
def Stratified_On(data, remove_columns, add_columns):
    #Removal of the columns - Profile phone and SignUp date 
    data = data.drop(['profile_phone', 'Signup_date'],axis=1)
    Stratified_on = list(data.columns.copy()) 
    #The removal of columns - (Etb, Ntb, Ptb) and (Upi_Flag, Ppi_Flag, Bbps_Flag) replaced with Customer Type and Payments active respectively 
    res = list(set(Stratified_on) - set(remove_columns)) 
    final_stratified_on = res+ add_columns
    new_features =[]
    for feature in final_stratified_on:
        new_feature={}
        new_feature = {'key': feature, 'value': False}
        new_features.append(new_feature)
    return new_features

# global selected_features 
#To call the features function and store in this variable - Features 
Features = features(data_Original,selected_features)
# To save the array of columns to be remove in the Stratified on columns and store in the variable - remove_columns 
remove_columns = ['Etb', 'Ntb','Ptb', 'Upi_Flag','Ppi_Flag','Bbps_Flag']
# To save the array of columns to be added in the Stratified on columns and store in the variable - add_columns 
add_columns = ['Customer Type', 'Payments Active']
#To call the Stratified_On function and store in this variable - Stratification_columns
Stratification_columns = Stratified_On(data_Original, remove_columns, add_columns)

#To update the master file using the details from the global array - Array_output_file  
def excel_update(array_output_final): 
    file = (str(THIS_FOLDER)+"\\Master_Sheet_V2.csv")
    df = pd.read_csv(file, on_bad_lines=None)
    #To have the variable index_len for the index number in the Master Sheet excel 
    index_len = len(df)+1
    array_output_local = array_output_final.copy()
    array_output_local.insert(0,index_len)
    with open(file, 'a' , newline ='') as f_object:
          writer_object = writer(f_object)
          writer_object.writerow(array_output_local)
          f_object.close()

#To have a function for calling all the parameters/input spaces in the form 
def get_form_parameters():
    Form  = {}
    if (request.form.get("Edit_button") != None) : 
            Form["Download"] = False
    Form['Hypothesis'] = request.form['Hypothesis'] if (request.form.get('Hypothesis') != None) else forms['Hypothesis']
    Form['Experiment'] = request.form['Experiment_input'] if (request.form.get('Experiment_input') != None) else forms['Experiment']
    Form['ConversionInterval'] = request.form['confidence_Interval_Input'] if (request.form.get('confidence_Interval_Input') != None) else forms['ConversionInterval']
    Form["MarginError"] = request.form['margin_Error_Input'] if (request.form.get('margin_Error_Input') != None) else forms['MarginError']
    Form['BaselineRate'] = request.form['baseline_Rate_Input'] if (request.form.get('baseline_Rate_Input') != None) else forms['BaselineRate']
    Form['DetectableEffect'] = request.form['detectable_Effect_Input'] if (request.form.get('detectable_Effect_Input') != None) else forms['DetectableEffect']
    Form['SignificantPower'] = request.form['significance_Power_Input'] if (request.form.get('significance_Power_Input') != None) else forms['SignificantPower']
    Form['SignificantLevel'] = request.form['significance_Level_Input'] if (request.form.get('significance_Level_Input') != None) else forms['SignificantLevel']
    Form["Total_cases"] = int(Form['Experiment'])+1
    Form['Operator'] = request.form['operator'] if (request.form.get('operator') != None) else forms['Operator']
    Form['CampaignName'] = request.form['campaign_name_input'] if (request.form.get('campaign_name_input') != None) else forms['CampaignName']
    Form['CampaignStartdate'] = request.form['campaign_start_date'] if (request.form.get('campaign_start_date') != None) else forms['CampaignStartdate']
    Form['CampaignEnddate'] = request.form['campaign_end_date'] if (request.form.get('campaign_end_date') != None) else forms['CampaignEnddate']
    Form['CampaignType'] = request.form['Campaign_type_input'] if (request.form.get('Campaign_type_input') != None) else 'Push'
    Form['ConversionMetric'] = request.form['Conversion_metric_input'] if (request.form.get('Conversion_metric_input') != None) else 'Retention rate'
    Form['ConversionPeriod'] = request.form['Conversion_period_input'] if (request.form.get('Conversion_period_input') != None) else forms['ConversionPeriod']
    Form['Sum'] = 0

    Form['evan_millers'] = 0
    Form['final_result'] = 0
    Form['basic_result'] = 0
    Form['Records_Available'] = 0 
    Form["Sampling_Result"] =""
    Form["Download"] = False



    # To save the details regarding the button clicked and the ID where it's supposed to get scrolled to 
    Button_Section = {"Sample_Size_submit_1":"sample_size", "Sample_Size_submit_2":"sample_size","Features_button":"select_filters","Sampling_Technique_submit":"sampling_technique","Final_submit":"campaign_details","Random_button":"sampling_technique","Test_cases_button":"Test_cases","stratify_button":"sampling_technique","Button_id":"sampling_technique", "Edit_button":"hypothesis_section"}
    for key,value in Button_Section.items(): 
        if (request.form.get(key) != None) :
            #Location key in the form is the ID where it's supposed to get scrolled to  
            Form['location'] = value
    return Form  


def sample_suggest(): 
    return "Systematic Sampling"   

#To verify if with the chosen sample size, number of experiment and current count it is possible to do the furture calculations 
def verification_func(doe, sample_size, current_count):
   if ((int(doe)+1)*int(sample_size) > int(current_count)):
       return "True"
   else : 
       return "False"
   
#To calculate how much of data base is required for the selected experiment number and minimum sample Size 
# Called when there is an error   
def requiredDB(doe, sample_size):
    return (int(doe)+1)*int(sample_size)

#To Efficiently return the count of the selected features 
#This function only returns the value 
def db_count(data,selected): 
    try:
        #The sum of the selected columns are counts and checked if it matches that of the 
        return (data[selected].sum(axis=1)).value_counts()[len(selected)]
    except:
        return 0 
    
def Output_file(data_list):
    try :    
    
        with pd.ExcelWriter(str(THIS_FOLDER)+"\\output.xlsx") as writer:
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
        data.index.name = "Index"
        try : 
             for i in range(0, Total): 
                 step = (len(data)/test_size[i])
                 indexes = np.arange(0, len(data)-1, step=step)
                 systematic_sample = data.iloc[indexes]
                 data.drop(systematic_sample.index, axis=0,inplace=True)
                 Test.append(systematic_sample)
             if (len(data) > threshold_control):
                 step = (len(data)/threshold_control)
                 indexes = np.arange(0, len(data)-1, step=step)
                 systematic_sample = data.iloc[indexes]
                 Test.append(systematic_sample)
             else : 
                 Test.append(systematic_sample) 
             if (Output_file(Test) ):
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
    data.index.name = "Index"
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
    data.index.name = "Index"
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
            # return "Reduce the number of Experiments to atmost " + str(loop_flag) +" or change the columns to be stratified on "
            return "Reduce the number of Experiments or change the columns to be stratified on "
        else : 
            return "The least populated class in y has only 1 member. Please change the columns to be stratified on"

#To return the base data.
# This is called after the Sampling Technique is chosen 
# The result of the number of base data is a different function.      
def base_data(data,Selected_DataBase): 
        for i in Selected_DataBase: 
             data = data[(data[i])==1]
        return data

#To compare the results of both the sample sizes and returns the larger one 
def compare_size(evan_miller,basic_result): 
    return evan_miller if evan_miller >= basic_result else basic_result

#To return the result of basic sample size 
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

#To result the result of Evan Miller Sample Size 
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


@Flask_App.route('/',methods=['GET' , "POST"])
def New_Testing():
    Section_open = 0
    verification = ""
    test_size = []
    
    if request.method == 'POST':
       global forms 
       forms = get_form_parameters()  
       forms["Strat_result"] = ''

       global selected_features
       global Features

       global array_output_final  
       global selected_columns
       global Sub_Campaign_Names

      

     
       for i in Features: 
            if ((request.form.get(i['key'])) or ((i['key'] in selected_features) and (request.form.get("Edit_button") != None))):
                i['value'] = True
                selected_features.append(i['key'])
            else : 
                i ['value'] = False 
                if ((i['key'] in selected_features) and (request.form.get("Edit_button") == None) and (request.form.get("Download_button") == None)): 
                    selected_features.remove(i['key'])
       selected_features = list(set(selected_features))
       
       for i in Stratification_columns: 
            variable = "Strat" + i['key']
            if ((request.form.get(variable)) or ((i['key'] in selected_columns) and  (request.form.get("Edit_button") != None))) :
                i['value'] = True
            else : 
                i ['value'] = False 

       if (request.form.get("Reset_button") != None):
          test_size = []
          selected_features = []
          selected_columns = []
          array_output_final = []
          Sub_Campaign_Names = []
          Section_open = 0
          forms = {}
          verification = ""
          forms = {'Total_cases': 2, 'Experiment': '1', 'ConversionInterval':'' , 'MarginError':'' , 'BaselineRate':'' , 'DetectableEffect':'' , 'SignificantPower':'' , 'SignificantLevel':'' ,'Campaign_Name_1' : '', 'Campaign_Name_2': '', 'Operator' : 'Default', 'Hypothesis':'', 'CampaignName' : '','CampaignStartdate' : '','CampaignEnddate' : '', 'CampaignType':'Push' , 'ConversionMetric':'Retention Rate' ,'ConversionPeriod' : ''}
          forms['Sum'] = 0
          forms['evan_millers'] = 0
          forms['final_result'] = 0
          forms['basic_result'] = 0
          forms['Records_Available'] = 0 
          forms["Sampling_Result"] =""
          forms["Download"] = False
      
       if (forms["ConversionInterval"] != '' and forms["MarginError"] != ''):   
            forms ["basic_result"] = basic_Sample_Size(forms["ConversionInterval"], forms["MarginError"])
            forms["final_result"] = compare_size(forms["evan_millers"],forms["basic_result"])
               
       if (forms["BaselineRate"] != '' and forms["DetectableEffect"] != '' and forms["SignificantPower"] != '' and forms["SignificantLevel"] != ''):           
           forms ["evan_millers"] =  evan_Millers(forms["BaselineRate"], forms["DetectableEffect"], forms["SignificantPower"], forms["SignificantLevel"])
           forms["final_result"] = compare_size(forms["evan_millers"],forms["basic_result"])
       
    #  global test_size 
       for i in range(1,int(forms["Experiment"])+1):
             variable = "Test_case_" + str(i)
             #To make sure that a blank values isn't submitted to the Test_Case inputs 
             if (request.form.get(variable) != None) : 
                 if (int(request.form[variable]) < int(forms['final_result'])):
                      forms[variable] = forms['final_result']
                      forms['Sum'] += int(forms[variable])
                 else :     
                    #If it's not empty 
                    forms[variable] = request.form[variable] 
                    forms['Sum'] += int(forms[variable])
             else : 
                # If it's empty 
                forms[variable] = forms['final_result']
                if (len(array_output_final) > 0):
                     forms[variable] = array_output_final[4][i-1] 
       
       
       
       if (len(Sub_Campaign_Names) != int(forms['Experiment'])): 
            Sub_Campaign_Names = [""]*int(forms['Experiment'])
       
       if (int(forms['Experiment']) == 1): 
            Sub_Campaign_Names[0]= forms['CampaignName']    

       for i in range(1,int(forms["Experiment"])+1):
             variable = "Campaign_Name_" + str(i)
             forms[variable] = ''
             if (request.form.get(variable) != None) : 
                    forms[variable] = request.form[variable] 
                    Sub_Campaign_Names[i-1] = forms[variable]
             else : 
                    forms[variable] = Sub_Campaign_Names[i-1]

       base_data_input =  base_data(data_Original, selected_features) 
       forms["Current_Count"] = db_count(data_Original, selected_features)
      
       dummy_test_size =[]
       for i in range(0,int(forms["Total_cases"])-1):
                variable = "Test_case_" + str(i+1)
                dummy_test_size.insert((i+1), int(forms[variable]))
       test_size = dummy_test_size

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
                selected_columns = list(set(selected_columns))   
                if (len(selected_columns) == 0):
                    forms["Error"] = "Stratification_columns"
                else : 
                    forms["Sampling_Result"] = Stratification_result(base_data_input, test_size,selected_columns, int(forms['final_result']))   

       if (len(selected_features)!= 0): 
           Section_open = 1
       else : 
           Section_open = 0
           forms["Feature_Error"] = "Feature_Error"

       global button_variable 
       if (request.form.get('Button_id') != None or button_variable == "True"):
           button_variable = "True"
           forms["Sample_suggestion"] = sample_suggest()
        
       for i in range(1,int(forms["Experiment"])+1):
            variable = "Test_case_" + str(i)
            if (len(array_output_final) > 0):
                     forms[variable] = array_output_final[4][i-1] 
                     forms["Sum"] += forms["final_result"] 
            elif(request.form.get(variable) == None or request.form.get(variable) == '' or int(request.form.get(variable))< int(forms["final_result"])):
                forms[variable] = forms["final_result"]
                forms["Sum"] += forms["final_result"] 
                        
       if (forms['final_result'] != 0 and forms['Current_Count'] != '' and len(selected_features)!= 0 ):
            # Records allocation opens up
            Section_open = 2
            verification = verification_func(forms["Experiment"], forms["final_result"], forms["Current_Count"])
            forms["required_db"] = requiredDB(forms["Experiment"],forms["final_result"])
            if (verification == "False" ):
                    forms["Records_Available"] = forms['Current_Count'] - forms['final_result'] - forms['Sum']
                    if (forms["Records_Available"] < 0): 
                        forms["Records_Available"] = "Please Reduce the number of record by " + str(abs(forms["Records_Available"]))
                        forms["Error"] ="Error-lessRecords"
                    else : 
                         forms["Records_Available"] = "Records Available for Allocation : " + str(forms['Current_Count'] - forms['final_result'] - forms['Sum'])
                
                         global Test_case_button
                         if ((request.form.get('Test_cases_button') != None) or (Test_case_button == True)): 
                             Section_open = 3
                             Test_case_button = True                
            if(verification == "True") : 
                forms['location'] = "modal_message"
   
       if (request.form.get("Final_submit") != None): 
        if (forms["CampaignName"] != '' and forms["CampaignStartdate"] != '' and forms["CampaignEnddate"] != '' and forms["CampaignType"] != '' and forms["ConversionMetric"] != '' and forms["ConversionPeriod"] != ''):           
                forms["Campaign_Details_Error"] = ""
                if ((len(Sub_Campaign_Names) != int(forms["Experiment"]) and (int(forms["Experiment"]) != 1)) or ('' in Sub_Campaign_Names)): 
                    forms["Campaign_Details_Error"] = "Please fill in all the Sub Campaign Names"
                if (datetime.strptime(forms["CampaignStartdate"], '%Y-%m-%d') > datetime.strptime(forms["CampaignEnddate"], '%Y-%m-%d')):
                    forms["Campaign_Details_Error"] = "The Campaign End Date should be after the Campaign Start Date"
                if (forms["Campaign_Details_Error"] == ""):    
                        array_output_final = [forms["Hypothesis"],forms['Experiment'],selected_features,forms["final_result"],test_size,forms["Operator"],selected_columns,forms["CampaignName"], Sub_Campaign_Names, today, forms['CampaignStartdate'], forms['CampaignEnddate'], forms['CampaignType'], forms['ConversionMetric'], forms['ConversionPeriod']]
                        forms ["Download"] = True  
        else : 
                forms["Campaign_Details_Error"] = "Please fill the form below completely"        

    #    if (forms["CampaignName"] != '' and forms["CampaignStartdate"] != '' and forms["CampaignEnddate"] != '' and forms["CampaignType"] != '' and forms["ConversionMetric"] != '' and forms["ConversionPeriod"] != '' and request.form.get("Final_submit") != None):           
    
    #         if ((len(Sub_Campaign_Names) != int(forms["Experiment"]) and (int(forms["Experiment"]) != 1)) or ('' in Sub_Campaign_Names)): 
    #             forms["Campaign_Details_Error"] = "Please fill in all the Sub Campaign Names."
    #         if (forms['CampaignName'] == '' or forms['CampaignStartdate'] == '' or forms['CampaignEnddate'] == '' or forms['ConversionPeriod'] == ''): 
    #             forms["Campaign_Details_Error"] = "Please fill all the details to proceed."
    #         if (datetime.strptime(forms["CampaignStartdate"], '%Y-%m-%d') > datetime.strptime(forms["CampaignEnddate"], '%Y-%m-%d')):
    #            forms["Campaign_Details_Error"] = "The campaign end date should be after the campaign start date"
    #         if (forms["Campaign_Details_Error"] == ""):    
    #                 array_output_final = [forms["Hypothesis"],forms['Experiment'],selected_features,forms["final_result"],test_size,forms["Operator"],selected_columns,forms["CampaignName"], Sub_Campaign_Names, today, forms['CampaignStartdate'], forms['CampaignEnddate'], forms['CampaignType'], forms['ConversionMetric'], forms['ConversionPeriod']]
    #                 forms ["Download"] = True 
              
 
            
       if (forms["Sampling_Result"] == "Sampling Successful" and len(selected_features)!= 0):
                #Campaign Details section opens up 
                Section_open = 4
       
       if (request.form.get("Download_button") != None): 
            try : 
                 
                 excel_update(array_output_final)
                 path = (str(THIS_FOLDER)+"\\output.xlsx")
                 forms ["Download"] = True 
                 return send_file(path, as_attachment=True)
            except : 
                  #One of the times this error occurs is when the Exc
                  forms["ExcelUpdate"] = "There is a problem while updating the Excel. Please Try again."
                  forms ["Download"] = True 

       
          

       return render_template('New_Testing.html', test_size = test_size, Stratification_columns = Stratification_columns, selected_features = selected_features, array_output_final = array_output_final,  Sub_Campaign_Names = Sub_Campaign_Names, Section_open = Section_open, stratification_columns = Stratification_columns, selected_columns =selected_columns, sum = sum, Success = True, form = forms, features = Features, date = today, Verification =  verification) 
    return render_template('New_Testing.html' , array_output_final = array_output_final, Section_open = Section_open, features = Features, Stratification_columns=Stratification_columns, date = today, form = forms)

#To display the intial page 
@Flask_App.route('/Index', methods=['GET','POST'])
def index():
    
    #It returns all the unique Campaign Names 
    def Names(data):
            return data['Master_campaign_name'].unique()

    #To create a main reference table 
    def create_main_table(choosen_Master_name, data) : 
        Sub_data = data[data['Master_campaign_name'] == choosen_Master_name]
        Sub_data = Sub_data.fillna(0)
        row = []
        For_rows = Sub_data[['group', 'sent_flag','campaign']].values.tolist()
        for i in For_rows: 
            if (i[0].lower() == 'control'):
                row.append('Control_base')
            elif (i[0].lower() == 'test' and i[1] == 1.0):
                row.append(i[2])
            elif (i[0].lower() == 'test' and i[1] == 0.0):
                row.append('Unsent_Test_base')
        Sub_data["X_axis_variables"] = row
        Sub_data['converted_percent'] = np.where(Sub_data['converted'] > 0, round(((Sub_data['converted']/Sub_data['users']))*100,2), 0)
        Sub_data['app_active_percent'] = np.where(Sub_data['app_active'] > 0, round(((Sub_data['app_active']/Sub_data['users']))*100,2), 0)   
        Sub_data['app_launched_percent'] = np.where(Sub_data['app_launched'] > 0, round(((Sub_data['app_launched']/Sub_data['users']))*100,2), 0)

        Sub_data = Sub_data.drop(['campaign', 'conversion_event', 'group', 'sent_flag', 'Master_campaign_name'], axis=1)
        Sub_data = Sub_data.sort_values("X_axis_variables") 
        return Sub_data
    
    #To get the values from the main_table to use in Series for the calculation
    def get_from_maintable (Main_table,Customer_type, percentage_what):
        Sub_data = Main_table[Main_table['customer_segment'] == Customer_type]
        Sub_data = Sub_data[['X_axis_variables',percentage_what, 'customer_segment']]
        Sub_data = Sub_data.sort_values("X_axis_variables") 
        return Sub_data[percentage_what].tolist()
    

   #To return the total percentage of the table for the line graph calculation 
    def Total_percentage(Main_table):
            Main_table
            Values = Main_table['X_axis_variables'].unique()
            Values = Values.tolist()
            Total = {}
            for i in Values : 
                Sub_data_2 = Main_table[Main_table['X_axis_variables'] == i]
                converted_Total = round((sum(Sub_data_2['converted'])/sum(Sub_data_2['users']))*100,2) 
                app_launch_Total = round((sum(Sub_data_2['app_launched'])/sum(Sub_data_2['users']))*100,2) 
                app_active_Total = round((sum(Sub_data_2['app_active'])/sum(Sub_data_2['users']))*100,2) 
                Total[i] = [converted_Total, app_launch_Total, app_active_Total] 
                Total_dataframe = pd.DataFrame.from_dict([Total])
                Total_dataframe = Total_dataframe.T
                Total_dataframe = Total_dataframe.reset_index()
                Total_dataframe.columns = ['Campaign_Reference', 'Total_percent']
                New_dataframe = pd.DataFrame(Total_dataframe['Total_percent'].to_list(), columns=['converted_percent','app_launched_percent','app_active_percent'])
                Total_dataframe = Total_dataframe['Campaign_Reference']
                final_output = pd.concat([ New_dataframe, Total_dataframe], axis=1)   
            return final_output

    #To create the series that will be used for the line graph + column graphs 
    def get_series (Main_table) :    
        list_MainFrame = [] 
        Dictionary_MainFrame ={}
        list_columns = ['converted_percent','app_active_percent','app_launched_percent']
        for i in range(len(list_columns)) : 
            list_MainFrame= [{'name' : 'Total', 'type' : 'line', 'data': Total_percentage(Main_table)[list_columns[i]].tolist()}]
            list_MainFrame.append({'name' : 'ETB', 'type' : 'bar', 'data': get_from_maintable(Main_table,'ETB', list_columns[i])})
            list_MainFrame.append({'name' : 'NTB', 'type' : 'bar', 'data': get_from_maintable(Main_table,'NTB', list_columns[i])})
            list_MainFrame.append({'name' : 'PTB', 'type' : 'bar', 'data': get_from_maintable(Main_table,'PTB', list_columns[i])})
            Dictionary_MainFrame[str(list_columns[i])] = list_MainFrame
        return Dictionary_MainFrame 
    
    #To find the maximum value in the entire table 
    #This is a required value for the graph as it helps in the creation of the Y axis for the columns 
    def max_value_for_graph (notes):        
        end_list = []
        for i in (notes): 
            list_hopefull_last_one = []
            for j in range(len(notes)+1):
                try : 
                    Max_number = max(notes[i][j]['data'])
                except : 
                    Max_number = 0 
                list_hopefull_last_one.append(Max_number) 
            end_list.append(max(list_hopefull_last_one))
        return end_list 
    
    #To add the total values for each Sub_campaign_Name using this function 
    def Total_percentage_add(Main_table):
        Values = Main_table['X_axis_variables'].unique().tolist()
        #This if-else is there because in case there is only one customer segment (ETB, PTB,NTB) then we don't need an All segment. 
        if (len(Main_table['customer_segment'].unique().tolist()) > 1) :        
            Total_all = []
            for i in Values : 
                Sub_data_2 = Main_table[Main_table['X_axis_variables'] == i]
                Total = []
                Total_sent = sum((Sub_data_2['sent']))
                Total_clicks = sum((Sub_data_2['clicks']))
                Total_users = sum((Sub_data_2['users']))
                Total_converted = sum((Sub_data_2['converted']))
                Total_App_Launched = sum((Sub_data_2['app_launched']))
                Total_App_Active = sum((Sub_data_2['app_active']))
                Total_converted_percent = round((Total_converted/Total_users)*100,2)
                Total_App_Active_percent = round((Total_App_Active/Total_users)*100,2)
                Total_App_Launched_percent = round((Total_App_Launched/Total_users)*100,2)
                Conversion = statistics.mean(Sub_data_2['Conversion_sign'])
                Total_all.append(["ALL",Conversion, Total_sent, Total_clicks,  Total_users,Total_converted,Total_App_Launched, Total_App_Active,i,Total_converted_percent, Total_App_Active_percent, Total_App_Launched_percent,])
            Total_all = pd.DataFrame(Total_all)
            Total_all.columns = ['customer_segment','Conversion_sign','sent','clicks','users','converted','app_launched','app_active','X_axis_variables','converted_percent','app_active_percent','app_launched_percent']
            return Main_table.append(Total_all)
        else : 
             return Main_table
             
    #To return the Zscore and Pvalue when given the details of Test Conversion, Control Conversion, Test Number,Control Number
    def Z_Score_P_Value(TC, CC, TN, CN): 
            test_CR = (TC/TN)
            control_CR = (CC/CN)
            total_CR = ((TC+CC)/(TN+CN))
            Z_score_value = (test_CR - control_CR)/(math.sqrt(total_CR*(1-total_CR)*((1/TN) + (1/CN))))
            # Probability distribution formula used for the calculation of P-value 
            return [round(Z_score_value,4) , round(norm.sf(Z_score_value),4)]
    
    #To return a table with the Z-score and P-value for each row. 
    #this function makes use of the Z_Score_P_Value function for the calculation of values 
    def Get_Zscore_PValue(Main_table): 
        Main_dummy_table = pd.DataFrame()
        
        list_abc = Main_table['customer_segment'].unique()
        Main_table["Sub_Campaign_Name"] = Main_table['X_axis_variables']
        Sub = Main_table[['customer_segment', 'sent', 'clicks', 'users', 'converted',
            'app_launched', 'app_active', 'X_axis_variables', 'converted_percent',
            'app_active_percent', 'app_launched_percent',"Sub_Campaign_Name", 'Conversion_sign']]
        Sub["Zscore"] = ''
        Sub["Pvalue"] = ''
        Sub['Significance'] = ''

        for i in list_abc:
            Sub_data = Sub[Sub['customer_segment'] == i]
            Sub_data["Zscore"] = np.where((Sub_data['X_axis_variables'] == "Unsent_Test_base") | (Sub_data['X_axis_variables'] == "Control_base"), "NA", 0)
            Sub_data["Pvalue"] = np.where((Sub_data['X_axis_variables'] == "Unsent_Test_base") | (Sub_data['X_axis_variables'] == "Control_base"), "NA", 0)

            Sub_data = Sub_data.set_index(['X_axis_variables'])
            CC = Sub_data["converted"]["Control_base"]
            CN = Sub_data["users"]["Control_base"]

            for i in Sub_data.index:
                if (Sub_data['Zscore'][i] != "NA"):
                    Values = Z_Score_P_Value(Sub_data["converted"][i], CC, Sub_data["users"][i], CN)
                    Sub_data['Zscore'][i] =  Values[0]
                    # Sub_data['Pvalue'][i] =  Values[1]
                    # Sub_data['Pvalue'][i] =  round(abs(int(Sub_data['Conversion_sign'][i]) - (Values[1])),4)
                    Sub_data['Pvalue'][i] = round((Values[1]),4)
                    #Higher the better, P value 
                    #Lower the better, 1-P value 
                    if (Sub_data['Conversion_sign'][i]):
                         Sub_data['Pvalue'][i] =  round(abs(int(Sub_data['Conversion_sign'][i]) - (Values[1])),4)         
                    else : 
                         Sub_data['Pvalue'][i] = round((Values[1]),4)              
                    Sub_data['Significance'][i] = "Significant" if Sub_data['Pvalue'][i] < 0.05 else "Insignificant"  
            Main_dummy_table = pd.concat([Main_dummy_table,Sub_data])

        return Main_dummy_table.sort_values(["Sub_Campaign_Name",'customer_segment'])

    #To read the csv file with all the details about the result/reporting 
    #The file should be csv while using read_csv 
    data = pd.read_csv(Path(THIS_FOLDER) / "AB_testing_report_format_new-master-2.csv")

    
    #Master_names is the name of all the Campaign Names 
    Master_names = Names(data)
    Form_graph = {}
    Form_graph['Master_Campaign_Name'] = 'Default'
    Form_graph['Master_Campaign_Name'] = request.form['Master_name_input'] if (request.form.get('Master_name_input') != None) else Form_graph['Master_Campaign_Name']
    # Form_graph['Master_Campaign_Name'] = request.form['Master_name_input'] if (request.form.get('Master_name_input') != None) else 'Default'
    
    Row_list =[]
    if (Form_graph['Master_Campaign_Name'] != 'Default'):
         choosen_Master_name = Form_graph['Master_Campaign_Name']
         Main_table = create_main_table(choosen_Master_name, data)  #This line creates a sub table with the choosen Master Name       
         Series = get_series(Main_table) #This line returns the Series that can be used in the line+column graphs. All the line+column graph's series are gotten using this method  
         Max_values = max_value_for_graph(Series) #This line return the max value in each graph so that it can scale accordingly [Column graph]
         new_index = 1 #for the
         X_axis_names = Main_table['X_axis_variables'].unique().tolist() #To get the X-axis variable names for the X-axis 
         Main_table = Total_percentage_add(Main_table) #To add the total percentage values to the table 
         Main_table = Get_Zscore_PValue(Main_table) #To Get the Z_score and P_values for all the values other than 'Control Base' and 'Unsent Test data' 
         #To save the table in the Row_list for displaying purpose 
         for index, rows in Main_table.iterrows():         
            my_list =[new_index, rows.Sub_Campaign_Name, rows.customer_segment, rows.sent, rows.clicks, rows.users, rows.converted, rows.app_launched, rows.app_active,rows.converted_percent,rows.app_active_percent,rows.app_launched_percent,rows.Zscore, rows.Pvalue, rows.Significance]
            Row_list.append(my_list)
            new_index += 1
         
    else : 
    #To return blank values in case the user didn't choosen the Master Campaign Name yet 
         Series = ''
         Main_table = ''
         Max_values = ''
         Row_list = ''
         X_axis_names = ''

    #To render the index1 template with these values 
    return render_template('index1.html', Form_graph = Form_graph, X_axis_names = X_axis_names, Row_list = Row_list, Series = Series , Max_values = Max_values, Master_names = Master_names)  

#It will run on Port no - 5129. The number is flexible. 
# http://localhost:<PORT NO>
# Example : http://localhost:5129
if __name__ == '__main__':
    Flask_App.debug = True
    # Flask_App.run(port= 5129)   
    Flask_App.run(host="0.0.0.0", port=5129, debug=True)

    
