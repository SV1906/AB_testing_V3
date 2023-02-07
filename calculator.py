from flask import Flask, render_template, request
import math
import scipy.stats as stats 
from werkzeug.utils import secure_filename
import pandas as pd 

Flask_App = Flask(__name__) # Creating our Flask Instance



@Flask_App.route('/', methods=['GET'])
def index():
    """ Displays the index page accessible at '/' """

    return render_template('index1.html')

@Flask_App.route('/operation_result/', methods=['POST'])
def operation_result():
    """Route where we send calculator form input"""

    error = None
    result = None

    # request.form looks for:
    # html tags with matching "name= "
    first_input = request.form['Input1']  
    second_input = request.form['Input2']
    third_input = request.form['Input3']
    fourth_input = request.form['Input4']
  #  operation = request.form['operation']

    try:
        input1 = float(first_input)
        input2 = float(second_input)
        input3 = float(third_input)
        input4 = float(fourth_input)

         # On default, the operation on webpage is addition
      #  if operation == "+":
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
        result = n 

     #   else:
         #   operation = "*"
          #  result = input1 * input2

        return render_template(
            'index1.html',
            input1=input1,
            input2=input2,
        #    operation=operation,
            result=result,
            calculation_success=True
        )
        
    except ZeroDivisionError:
        return render_template(
            'index1.html',
            input1=input1,
            input2=input2,
       #     operation=operation,
            result="Bad Input",
            calculation_success=False,
            error="You cannot divide by zero"
        )
        
    except ValueError:
        return render_template(
            'index1.html',
            input1=first_input,
            input2=second_input,
        #    operation=operation,
            result="Bad Input",
            calculation_success=False,
            error="Cannot perform numeric operations with provided input"
        )


@Flask_App.route('/Calculator/', methods=['POST'])
def Calculator():
    """Route where we send calculator form input"""

    error = None
    result = None

    # request.form looks for:
    # html tags with matching "name= "
    Error = request.form['Error']  
    Std_Deviation = request.form['Std deviation']
    Z_value = request.form['Z-value']
  #  operation = request.form['operation']

    try:
        input1 = float(Error)
        input2 = float(Std_Deviation)
        input3 = float(Z_value)

         # On default, the operation on webpage is addition
      #  if operation == "+":
        result = input3 

     #   else:
         #   operation = "*"
          #  result = input1 * input2

        return render_template(
            'index1.html',
            input1=input1,
            input2=input2,
        #    operation=operation,
            result=result,
            calculation_success=True
        )
        
    except ZeroDivisionError:
        return render_template(
            'index1.html',
            input1=input1,
            input2=input2,
       #     operation=operation,
            result="Bad Input",
            calculation_success=False,
            error="You cannot divide by zero"
        )

@Flask_App.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        sample_size = None 
        f = request.files['file']
        f.save(f.filename)  
      #  data = pd.read_csv(f)
      #  sample_size = data.shape[0]
      #  print(sample_size)
        return render_template("index1.html", name = f.filename)  
        

if __name__ == '__main__':
    Flask_App.debug = True
    Flask_App.run(host='0.0.0.0', port=400)
