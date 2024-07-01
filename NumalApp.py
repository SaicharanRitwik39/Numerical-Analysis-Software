#Importing the necessary libraries. Lines 2-15.
import math
import time
import sympy
import base64
import requests
import numpy as np
import pandas as pd
from sympy import *
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from streamlit_lottie import st_lottie


#Function used for calling the Lottie files. Lines 18-23.
@st.cache_data()
def load_lottieurl(url:str):
    r = requests.get(url)
    if r.status_code != 200:
       return None
    return r.json()


#Function used for generating DOWNLOAD LINKS. Lines 27-34.
def df_to_link(df, title='Download csv', filename='download.csv'):
    """Generates a link allowing the data in a given pandas dataframe to be downloaded.
    input:  dataframe
    output: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{title}</a>'


#Page configuration settings. Lines 38-41.
st.set_page_config(
    page_title = 'Numerical Analysis',
    page_icon = '💹'
)



#TITLE and Lottie animation code. Lines 46-54.
col1, col2 = st.columns([2,1])
with col1:
     st.title("NUMERICAL ANALYSIS") 
     st.write("This app will help you with numerical analysis.")  
with col2:
     url1 = "https://assets5.lottiefiles.com/packages/lf20_ax16hsjz.json"
     res1_json = load_lottieurl(url1)
     st_lottie(res1_json)
st.write("***")        



#SIDEBAR information Lines 59-75.
with st.sidebar:
    options = st.selectbox(
     'Choose amongst the following numerical methods:',
     ('Root Finding Techniques', 'Numerical Integration'))
    
    if (options == "Root Finding Techniques"):
       optionsRFT = st.radio(
           'Choose a Root Finding Technique:',
           ('Bisection Method', 'False Point Method'))
    elif (options == "Numerical Integration"):
        optionsNI = st.radio(
           'Choose a Numerical Integration Method:',
           ('Basic Trapezoidal Rule', 'Basic Simpson\'s 1/3 Rule', '2 Points Gauss Legendre Method', '3 Points Gauss Legendre Method'))
    



#BISECTION METHOD code. lines 80-262.
if ( options == "Root Finding Techniques" and optionsRFT == "Bisection Method"):
   col1, col2 = st.columns([2,1])
   with col1:
        st.subheader("BISECTION METHOD")
        st.info("By default, the tolerance is taken to be 0.000001.")   
   with col2:
        url2 = "https://assets1.lottiefiles.com/private_files/lf30_hqvoshsq.json"
        res2_json = load_lottieurl(url2)
        st_lottie(res2_json) 
   with st.form(key = "formbisection"): 
        s = st.text_input("Enter your function ( f(x) ):", value = "x**3-2*x-5")
        st.write("The function is:")
        st.write(sympy.sympify(s))
        col1, col2 = st.columns(2)
        with col1:
             leftside = st.number_input("Enter left side of the interval:", value = 2.0, format = "%.3f")
        with col2:
             rightside = st.number_input("Enter right side of the interval:", value = 3.0, format = "%.3f")
                
        submitbisection = st.form_submit_button("Submit")        
         
        
   with st.form(key = "formbisectionstoppingcrit"):
        st.write("Choose the stopping criteria:")        
        tolerancechoice = st.checkbox('Tolerance')
        tolerance = st.number_input("Enter the tolerance:", value = 0.000001, format = "%.6f")
        iterationschoice = st.checkbox('Iterations')
        iterations = st.number_input("Enter the number of iterations:", min_value = 1, value = 5, step = 1)
        
        submitbisectionstoppingcriteria = st.form_submit_button("Submit")
        
   np.arrayleftsidedata = []
   np.arrayrightsidedata = []
   np.arrayc = []
   np.arrayfunctioneval = [] 
    

#Code for bisection method if user only wants to enter the TOLERANCE. Lines 118-151.    
   if (tolerancechoice == True and iterationschoice == False):
      error = abs(rightside-leftside) 
      count = 0 
      while(error > tolerance):
           c = (leftside+rightside)/2
        
           if ( (sympy.sympify(s).subs({"x":float(leftside)}))*(sympy.sympify(s).subs({"x":float(rightside)})) > 0 ):
              st.warning('f(a) and f(b) must have different signs!')
              quit() 
                
           elif (sympy.sympify(s).subs({"x":float(leftside)}) == 0 or sympy.sympify(s).subs({"x":float(rightside)}) == 0):
              st.write('You already have the root!')
            
           elif ( float(sympy.sympify(s).subs({"x":float(leftside)}))*(sympy.sympify(s).subs({"x":float(c)})) < 0 ):
              rightside = c
              error = abs(rightside-leftside)
              count+=1 
                
           elif ( (sympy.sympify(s).subs({"x":float(rightside)}))*(sympy.sympify(s).subs({"x":float(c)})) < 0 ):
              leftside = c
              error = abs(rightside-leftside)
              count+=1  
                
           else: 
              st.error("Something went wrong!")
              quit() 
           np.arrayleftsidedata.append(leftside)
           np.arrayrightsidedata.append(rightside)
           np.arrayc.append(c)
           np.arrayfunctioneval.append(sympy.sympify(s).subs({"x":float(c)}))
       
      
      dfleftsidedata = pd.DataFrame(np.arrayleftsidedata, columns = ['Left Side of the Interval']) 
      dfrightsidedata = pd.DataFrame(np.arrayrightsidedata, columns = ['Right Side of the Interval'])
      dffunctionvalueatc = pd.DataFrame(np.arrayfunctioneval, columns = ['f(Midpoint)'])  
      dfc = pd.DataFrame(np.arrayc, columns = ['Mid Point'])  
      df = pd.concat([dfleftsidedata, dfrightsidedata, dfc, dffunctionvalueatc], axis = 1) 
     
            

#Code for BISECTION METHOD if user only wants the DEFAULT settings. Lines 156-189.             
   elif (tolerancechoice == False and iterationschoice == False):
        error = abs(rightside-leftside)
        count = 0
        while(error > 0.000001):
             c = (leftside+rightside)/2
            
             if ( (sympy.sympify(s).subs({"x":float(leftside)}))*(sympy.sympify(s).subs({"x":float(rightside)})) > 0 ):
                st.warning('f(a) and f(b) must have different signs!')
                quit()
                
             #elif (sympy.sympify(s).subs({"x":float(leftside)}) == 0 or sympy.sympify(s).subs({"x":float(rightside)}) == 0):
              #st.write('You already have the root!')
              #break 
                    
             elif ( float(sympy.sympify(s).subs({"x":float(leftside)}))*(sympy.sympify(s).subs({"x":float(c)})) < 0 ):
                rightside = c
                error = abs(rightside-leftside)
                count+=1 
                    
             elif ( (sympy.sympify(s).subs({"x":float(rightside)}))*(sympy.sympify(s).subs({"x":float(c)})) < 0 ):
                leftside = c
                error = abs(rightside-leftside)
                count+=1 
                    
             else: 
                st.error("Something went wrong!")
                quit() 
             np.arrayleftsidedata.append(leftside)
             np.arrayrightsidedata.append(rightside)
             np.arrayc.append(c)
             np.arrayfunctioneval.append(sympy.sympify(s).subs({"x":float(c)}))
       
      
        dfleftsidedata = pd.DataFrame(np.arrayleftsidedata, columns = ['Left Side of the Interval']) 
        dfrightsidedata = pd.DataFrame(np.arrayrightsidedata, columns = ['Right Side of the Interval'])
        dffunctionvalueatc = pd.DataFrame(np.arrayfunctioneval, columns = ['f(Midpoint)'])  
        dfc = pd.DataFrame(np.arrayc, columns = ['Mid Point'])  
        df = pd.concat([dfleftsidedata, dfrightsidedata, dfc, dffunctionvalueatc], axis = 1)    
                

#Code for bisection method if user only wants just the ITERATIONS. Lines 141-163.    
   elif (tolerancechoice == False and iterationschoice == True): 
        error = abs(rightside-leftside)
        count = 0
        while(count <= iterations):
             c = (leftside+rightside)/2
                
             if ( (sympy.sympify(s).subs({"x":float(leftside)}))*(sympy.sympify(s).subs({"x":float(rightside)})) > 0 ):
                st.warning('f(a) and f(b) must have different signs!')
                quit()
                
             elif ( float(sympy.sympify(s).subs({"x":float(leftside)}))*(sympy.sympify(s).subs({"x":float(c)})) < 0 ):
                rightside = c
                error = abs(rightside-leftside)
                count+=1   
                
             elif ( (sympy.sympify(s).subs({"x":float(rightside)}))*(sympy.sympify(s).subs({"x":float(c)})) < 0 ):
                leftside = c
                error = abs(rightside-leftside)
                count+=1   
                
             else: 
                st.error("Something went wrong!")
                quit() 
                
                
#Code for bisection method if user wants to enter both the ITERATIONS and TOLERANCE. Lines 167-189. 
   elif (tolerancechoice == True and iterationschoice == True):
        error = abs(rightside-leftside)
        count = 0
        while(count <= iterations and error > tolerance):
             c = (leftside+rightside)/2
                
             if ( (sympy.sympify(s).subs({"x":float(leftside)}))*(sympy.sympify(s).subs({"x":float(rightside)})) > 0 ):
                st.warning('f(a) and f(b) must have different signs!')
                quit() 
                
             elif ( float(sympy.sympify(s).subs({"x":float(leftside)}))*(sympy.sympify(s).subs({"x":float(c)})) < 0 ):
                rightside = c
                error = abs(rightside-leftside)
                count+=1  
                
             elif ( (sympy.sympify(s).subs({"x":float(rightside)}))*(sympy.sympify(s).subs({"x":float(c)})) < 0 ):
                leftside = c
                error = abs(rightside-leftside)
                count+=1  
                
             else: 
                st.error("Something went wrong!")
                quit()    
                
                
   st.subheader("RESULTS")
   col1, col2, col3, col4 = st.columns(4)
   with col1:
        st.write("The error is:")     
        st.text(error)
   with col2:
        st.write("The root is:")
        st.text(c)
   with col3:
        st.write("Number of Iterations:")
        st.text(count-1)
   #with col4:
        #st.metric('Computation Time (sec):', elapsed_time)
    
   download_link = df_to_link(df, title='Iterations Data', filename='Iteration.csv')
   st.markdown(download_link, unsafe_allow_html=True)
   #st.dataframe(df) 
        
   st.write("***")     
          
        


        
#FALSE POINT METHOD code. Lines 213-.        
if ( options == "Root Finding Techniques" and optionsRFT == "False Point Method"):
   col1, col2 = st.columns([2,1])
   with col1:
        st.subheader("FALSE POSITION METHOD") 
   with col2:
        url2 = "https://assets6.lottiefiles.com/packages/lf20_8szshemh.json"
        res2_json = load_lottieurl(url2)
        st_lottie(res2_json)
   with st.form(key = "formfalsepoint"): 
        s = st.text_input("Enter your function ( f(x) ):", value = "2*E**(x)*sin(x)-3")
        if s == " ":
           st.warning("You have to enter something!") 
        st.write("The function is:")
        st.write(sympy.sympify(s))
        col1, col2 = st.columns(2)
        with col1:
             leftside = st.number_input("Enter left side of the interval:", value = 0.0, format = "%.3f")
        with col2:
             rightside = st.number_input("Enter right side of the interval:", value = 1.0, format = "%.3f") 
                
        submitfalsepoint = st.form_submit_button("Submit")  
        
    
   st.write("Choose the stopping criteria:")        
   tolerancechoice = st.checkbox('Tolerance')
   if tolerancechoice:
      tolerance = st.number_input("Enter the tolerance:", value = 0.00001, format = "%.6f")
   iterationschoice = st.checkbox('Iterations')
   if iterationschoice:
      iterations = st.number_input("Enter the number of iterations:", min_value = 1, value = 5, step = 1)
   st.write("***") 


#Code for FALSE POINT METHOD if user only wants to enter the TOLERANCE. Lines 249-.
   if (tolerancechoice == True and iterationschoice == False):
      count = 0
      c_before = 0
      c = ( leftside*(sympy.sympify(s).subs({"x":float(rightside)})) - rightside*(sympy.sympify(s).subs({"x":float(leftside)})) )/( sympy.sympify(s).subs({"x":float(rightside)}) - (sympy.sympify(s).subs({"x":float(leftside)})) ) 
      error = abs(c-c_before)
        
      while(error > tolerance):
           c_after = ( leftside*(sympy.sympify(s).subs({"x":float(rightside)})) - rightside*(sympy.sympify(s).subs({"x":float(leftside)})) )/( sympy.sympify(s).subs({"x":float(rightside)}) - (sympy.sympify(s).subs({"x":float(leftside)})) )
            
           if ( (sympy.sympify(s).subs({"x":float(leftside)}))*(sympy.sympify(s).subs({"x":float(rightside)})) >= 0 ):
               st.warning('f(a) and f(b) must have different signs!')
               quit()
            
           elif ( (sympy.sympify(s).subs({"x":float(c_after)}))*(sympy.sympify(s).subs({"x":float(leftside)})) < 0 ):
                 error = abs(c_after-rightside)  
                 rightside = c_after
                 count+=1
            
           elif ( (sympy.sympify(s).subs({"x":float(c_after)}))*(sympy.sympify(s).subs({"x":float(rightside)})) < 0 ): 
                error = abs(c_after-leftside)  
                leftside = c_after
                count+=1 
                
           else: 
                st.error("Something went wrong!")
                quit() 
                
                
#Code for FALSE POINT METHOD if user only wants the DEFAULT settings. Lines 280-302.             
   elif (tolerancechoice == False and iterationschoice == False):
        count = 0
        c_before = 0
        c = ( leftside*(sympy.sympify(s).subs({"x":float(rightside)})) - rightside*(sympy.sympify(s).subs({"x":float(leftside)})) )/( sympy.sympify(s).subs({"x":float(rightside)}) - (sympy.sympify(s).subs({"x":float(leftside)})) ) 
        error = abs(c-c_before)
        while(error > 0.000001):
             c_after = ( leftside*(sympy.sympify(s).subs({"x":float(rightside)})) - rightside*(sympy.sympify(s).subs({"x":float(leftside)})) )/( sympy.sympify(s).subs({"x":float(rightside)}) - (sympy.sympify(s).subs({"x":float(leftside)})) )
            
             if ( (sympy.sympify(s).subs({"x":float(leftside)}))*(sympy.sympify(s).subs({"x":float(rightside)})) >= 0 ):
                 st.warning('f(a) and f(b) must have different signs!')
                 quit()
            
             elif ( (sympy.sympify(s).subs({"x":float(c_after)}))*(sympy.sympify(s).subs({"x":float(leftside)})) < 0 ):
                 error = abs(c_after-rightside)  
                 rightside = c_after
                 count+=1
            
             elif ( (sympy.sympify(s).subs({"x":float(c_after)}))*(sympy.sympify(s).subs({"x":float(rightside)})) < 0 ): 
                error = abs(c_after-leftside)  
                leftside = c_after
                count+=1 
                
             else: 
                st.error("Something went wrong!")
                quit()  
                
                
#Code for FALSE POINT METHOD if user only wants the ITERATIONS. Lines 280-302.             
   elif (tolerancechoice == False and iterationschoice == True):
        count = 0
        c_before = 0
        c = ( leftside*(sympy.sympify(s).subs({"x":float(rightside)})) - rightside*(sympy.sympify(s).subs({"x":float(leftside)})) )/( sympy.sympify(s).subs({"x":float(rightside)}) - (sympy.sympify(s).subs({"x":float(leftside)})) ) 
        error = abs(c-c_before)
        while(count < iterations):
             c_after = ( leftside*(sympy.sympify(s).subs({"x":float(rightside)})) - rightside*(sympy.sympify(s).subs({"x":float(leftside)})) )/( sympy.sympify(s).subs({"x":float(rightside)}) - (sympy.sympify(s).subs({"x":float(leftside)})) )
            
             if ( (sympy.sympify(s).subs({"x":float(leftside)}))*(sympy.sympify(s).subs({"x":float(rightside)})) >= 0 ):
                 st.warning('f(a) and f(b) must have different signs!')
                 quit()
            
             elif ( (sympy.sympify(s).subs({"x":float(c_after)}))*(sympy.sympify(s).subs({"x":float(leftside)})) < 0 ):
                 error = abs(c_after-rightside)  
                 rightside = c_after
                 count+=1
            
             elif ( (sympy.sympify(s).subs({"x":float(c_after)}))*(sympy.sympify(s).subs({"x":float(rightside)})) < 0 ): 
                error = abs(c_after-leftside)  
                leftside = c_after
                count+=1 
                
             else: 
                st.error("Something went wrong!")
                quit()          
                
                
                
#Code for FALSE POINT METHOD if user wants both the ITERATIONS and TOLERANCE. Lines 337-361.             
   elif (tolerancechoice == True and iterationschoice == True):
        count = 0
        c_before = 0
        c = ( leftside*(sympy.sympify(s).subs({"x":float(rightside)})) - rightside*(sympy.sympify(s).subs({"x":float(leftside)})) )/( sympy.sympify(s).subs({"x":float(rightside)}) - (sympy.sympify(s).subs({"x":float(leftside)})) ) 
        error = abs(c-c_before)
        while(count <= iterations and error > tolerance):
             c_after = ( leftside*(sympy.sympify(s).subs({"x":float(rightside)})) - rightside*(sympy.sympify(s).subs({"x":float(leftside)})) )/( sympy.sympify(s).subs({"x":float(rightside)}) - (sympy.sympify(s).subs({"x":float(leftside)})) )
            
             if ( (sympy.sympify(s).subs({"x":float(leftside)}))*(sympy.sympify(s).subs({"x":float(rightside)})) >= 0 ):
                 st.warning('f(a) and f(b) must have different signs!')
                 quit()
            
             elif ( (sympy.sympify(s).subs({"x":float(c_after)}))*(sympy.sympify(s).subs({"x":float(leftside)})) < 0 ):
                 error = abs(c_after-rightside)  
                 rightside = c_after
                 count+=1
            
             elif ( (sympy.sympify(s).subs({"x":float(c_after)}))*(sympy.sympify(s).subs({"x":float(rightside)})) < 0 ): 
                error = abs(c_after-leftside)  
                leftside = c_after
                count+=1 
                
             else: 
                st.error("Something went wrong!")
                quit()                                 
                
                
   st.subheader("RESULTS")
   col1, col2, col3, col4 = st.columns(4)
   with col1:
        st.write("The error is:")     
        st.text(error)
   with col2:
        st.write("The root is:")
        st.text(float(c_after)) 
   with col3:
        st.write("Number of Iterations:")
        st.text(count) 
        
   st.write("***")

######################################################################################################################################  
                                                   #NUMERICAL INTEGRATION METHODS#      

###########################################################################################################################################

#TRAPEZOIDAL RULE CODE.
if (options == "Numerical Integration" and optionsNI == "Basic Trapezoidal Rule"):    
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("BASIC TRAPEZOIDAL RULE")   
    with col2:
        url2 = "https://assets7.lottiefiles.com/packages/lf20_fqmfqbvk.json"
        res2_json = load_lottieurl(url2)
        st_lottie(res2_json)
        
    with st.form(key = "formbasictrapezoidalrule"):
         col3, col4 = st.columns([2,1])
         s = st.text_input("Enter your function ( f(x) ):", value = "x**3-2*x-5")
         with col3:
              lowerlimit = st.number_input("Enter lower limit:", value = 2.0, format = "%.3f")
         with col4:
              upperlimit = st.number_input("Enter upper limit:", value = 3.0, format = "%.3f")
         st.write("The function is:")
         st.write(sympy.sympify(s))
         submitbasictrapezoidalrule = st.form_submit_button("Submit")
    st.subheader("Results")  
    st.write("The value of the integral is:")
    btrresult = (upperlimit-lowerlimit)*(sympy.sympify(s).subs({"x":float(lowerlimit)})+sympy.sympify(s).subs({"x":float(upperlimit)}))/2
    st.text(btrresult)
    

#BASIC SIMPSON 1/3 RULE CODE.    
elif(options == "Numerical Integration" and optionsNI == "Basic Simpson\'s 1/3 Rule"):
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("BASIC SIMPSON\'s 1/3 RULE")   
    with col2:
        url = "https://assets3.lottiefiles.com/packages/lf20_lglwitrl.json"
        res2_json = load_lottieurl(url)
        st_lottie(res2_json)
        
    with st.form(key = "basicsimpson13"):
         col3, col4 = st.columns([2,1])
         s = st.text_input("Enter your function ( f(x) ):", value = "x**3-2*x-5")
         with col3:
              lowerlimit = st.number_input("Enter lower limit:", value = 2.0, format = "%.3f")
         with col4:
              upperlimit = st.number_input("Enter upper limit:", value = 3.0, format = "%.3f")
         st.write("The function is:")
         st.write(sympy.sympify(s))
         
         h = (upperlimit-lowerlimit)/2
         basicsimpson13 = st.form_submit_button("Submit")  
         
    st.subheader("Results")
    st.write("The value of the integral is:")
    basicsimpson13result = (h/3)*(sympy.sympify(s).subs({"x":float(upperlimit)})+sympy.sympify(s).subs({"x":float(lowerlimit)})+sympy.sympify(s).subs({"x":float((upperlimit+lowerlimit)/2)}))
    st.text(basicsimpson13result)     
    
    
#2 POINTS GAUSS LEGENDRE METHOD CODE.
elif(options == "Numerical Integration" and optionsNI == "2 Points Gauss Legendre Method"):
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("2 POINTS GAUSS LEGENDRE METHOD")   
    with col2:
        url = "https://assets9.lottiefiles.com/temporary_files/5fDz2o.json"
        res2_json = load_lottieurl(url)
        st_lottie(res2_json)
        
    with st.form(key = "2pointsgausslegendre"):
         col3, col4 = st.columns([2,1])
         s = st.text_input("Enter your function ( f(x) ):", value = "x**3-2*x-5")
         with col3:
              lowerlimit = st.number_input("Enter lower limit:", value = 2.0, format = "%.3f")
         with col4:
              upperlimit = st.number_input("Enter upper limit:", value = 3.0, format = "%.3f")
         st.write("The function is:")
         st.write(sympy.sympify(s))
         
         x1 = ((upperlimit-lowerlimit)/2)*(-0.57735026919) + (upperlimit+lowerlimit)/2
         x2 = ((upperlimit-lowerlimit)/2)*(0.57735026919) + (upperlimit+lowerlimit)/2 
        
         points2gausslegendre = st.form_submit_button("Submit")  
         
    st.subheader("Results")
    gaussleg2result = ((upperlimit-lowerlimit)/2)*(sympy.sympify(s).subs({"x":float(x1)})+sympy.sympify(s).subs({"x":float(x2)}))
    st.write("The value of the integral is:")
    st.text(gaussleg2result)    
    
    
    
elif(options == "Numerical Integration" and optionsNI == "3 Points Gauss Legendre Method"):
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("3 POINTS GAUSS LEGENDRE METHOD")   
    with col2:
        url = "https://assets9.lottiefiles.com/temporary_files/a1Aw4V.json"
        res2_json = load_lottieurl(url)
        st_lottie(res2_json)
        
    with st.form(key = "3pointsgausslegendre"):
         col3, col4 = st.columns([2,1])
         s = st.text_input("Enter your function ( f(x) ):", value = "x**3-2*x-5")
         with col3:
              lowerlimit = st.number_input("Enter lower limit:", value = 2.0, format = "%.3f")
         with col4:
              upperlimit = st.number_input("Enter upper limit:", value = 3.0, format = "%.3f")
         st.write("The function is:")
         st.write(sympy.sympify(s))
         
         x1 = ((upperlimit-lowerlimit)/2)*(-0.7745966692) + (upperlimit+lowerlimit)/2
         x2 = (upperlimit+lowerlimit)/2
         x3 = ((upperlimit-lowerlimit)/2)*(0.7745966692) + (upperlimit+lowerlimit)/2 
        
         points2gausslegendre = st.form_submit_button("Submit")  
         
    st.subheader("Results")
    gaussleg2result = ((upperlimit-lowerlimit)/2)*((5/9)*(sympy.sympify(s).subs({"x":float(x1)}))+(8/9)*sympy.sympify(s).subs({"x":float(x2)})+(5/9)*(sympy.sympify(s).subs({"x":float(x3)})))
    st.write("The value of the integral is:")
    st.text(gaussleg2result)    