import os
import numpy as np
from pyzbar.pyzbar import decode
import sqlite3
import cv2
import onnxruntime as rt
import sys
from PySide2.QtWidgets import QCheckBox,QDialog,  QApplication, QMainWindow, QLabel,QMessageBox ,QVBoxLayout,QHBoxLayout, QWidget, QLineEdit, QProgressBar, QPushButton, QFileDialog, QListWidget, QComboBox
from PySide2.QtGui import QFont 
from PySide2.QtCore import Qt, QThread, Signal
import webbrowser
import json
import ctypes
import pandas as pd
from ultralytics import YOLO
import sys
import os
import webbrowser
import socket
import ctypes
import logging
import secrets
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from plotly.offline import plot
import plotly.graph_objs as go

from PySide2.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QTextEdit
from PySide2.QtCore import QThread, Signal
from flask import Flask, render_template, request, session, redirect, url_for, jsonify, send_file
from flask_session import Session
import re


import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")



# Setup Flask app
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Generates a secure random key
app.config['SESSION_TYPE'] = 'filesystem'  # Specify the session type
app.config['UPLOAD_FOLDER'] = 'static/plots/'

logging.basicConfig(level=logging.INFO)

logging.info("Starting the Flask application...")


# Create the Flask app


script_dir = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))

app = Flask(__name__, 
            template_folder=os.path.join(script_dir, 'templates'),
            static_folder=os.path.join(script_dir, 'static'))

app.secret_key = secrets.token_hex(16)  # Generates a secure random key
app.config['SESSION_TYPE'] = 'filesystem'  # Specify the session type
app.config['UPLOAD_FOLDER'] = 'static/plots/'
Session(app)  # Initialize the session
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'plots')

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


#onnx_model_path1_Q = os.path.join(script_dir, "Dependencies", "question.onnx")
onnx_model_path1_Q_40 = os.path.join(script_dir, "Dependencies","mq_40_50_70.onnx")# "question_40_50_70.onnx")
onnx_model_path1_Q_50 = os.path.join(script_dir, "Dependencies","mq_40_50_70.onnx") #"question_40_50_70.onnx")
onnx_model_path1_Q_70 = os.path.join(script_dir, "Dependencies","mq_40_50_70.onnx") #"question_40_50_70.onnx")




onnx_model_path1_Q_100 = os.path.join(script_dir, "Dependencies","model_q100_june28.onnx")#"Q100_Q150_model_v1.onnx")
onnx_model_path1_Q_150 = os.path.join(script_dir, "Dependencies", "prsu_150_model5_with_aug12.onnx")



onnx_model_course  = os.path.join(script_dir, "Dependencies","course_JULY11_1.onnx") #"course_gray33.onnx")
onnx_model_question_paper_Series_Code  = os.path.join(script_dir, "Dependencies", "series_gray33.onnx")
onnx_model_Roll1  = os.path.join(script_dir, "Dependencies","roll(0-9)_F2_3.onnx")# "roll(0-9)_gray.onnx")
#onnx_model_Roll1  = os.path.join(script_dir, "Dependencies", "roll(0-9)_gray.onnx") "roll(1-0).onnx")

onnx_model_Roll2  = os.path.join(script_dir, "Dependencies","Roll(1-0)_JULY13.onnx")

#onnx_model_sitting = os.path.join(script_dir, "Dependencies","sitting_sept26_2024.onnx")#"sitting_new.onnx")# "sitting_gray.onnx")

onnx_model_sitting = os.path.join(script_dir, "Dependencies","sitting_F2_1.onnx")
onnx_model_stream_horizontal = os.path.join(script_dir, "Dependencies","STREAM_ARTS_SCI_COMM_JULY11.onnx")#" "stream_horizontal_gray.onnx")
onnx_model_stream_vertical = os.path.join(script_dir, "Dependencies", "stream_vertical_gray1.onnx")
onnx_model_category1 = os.path.join(script_dir, "Dependencies", "category_bc_gen_sc_st_new1.onnx")#category_pattern1_gray4.onnx")



onnx_model_category2=os.path.join(script_dir, "Dependencies", "category(OBC_SC_ST_UR).onnx")


onnx_model_stream_sub_category=os.path.join(script_dir, "Dependencies", "Stream(sub_category).onnx")
onnx_model_booklet_code =os.path.join(script_dir, "Dependencies", "booklet_code_series.onnx")

onnx_model_domicle=os.path.join(script_dir, "Dependencies", "yes_no.onnx")


onnx_model_gender=os.path.join(script_dir, "Dependencies", "gender11.onnx")


onnx_model_disabled=os.path.join(script_dir, "Dependencies", "yes_no.onnx")

onnx_model_subject_group=os.path.join(script_dir, "Dependencies", "pcm_pcb11.onnx")

onnx_model_faculty=os.path.join(script_dir, "Dependencies", "faculty.onnx")
onnx_model_series=os.path.join(script_dir, "Dependencies", "series.onnx")
onnx_model_series2=os.path.join(script_dir, "Dependencies", "series_hor_f2.onnx")# "series2.onnx")

html_path=os.path.join(script_dir, "Dependencies","local_host11.html")

model_classification=os.path.join(script_dir, "Dependencies","omr_digit_yolo_classification.pt")

model2 = YOLO(model_classification)

# onnx_model_path =os.path.join(script_dir, "Dependencies","yolo_classification.onnx")

# model2 = rt.InferenceSession(onnx_model_path)

model_Q_40 = rt.InferenceSession(onnx_model_path1_Q_40 )
model_Q_50 = rt.InferenceSession(onnx_model_path1_Q_50 )
model_Q_70 = rt.InferenceSession(onnx_model_path1_Q_70 )
model_Q_100 = rt.InferenceSession(onnx_model_path1_Q_100 )
model_Q_150 = rt.InferenceSession(onnx_model_path1_Q_150 )

model_course = rt.InferenceSession(onnx_model_course)

model_question_paper_Series_Code = rt.InferenceSession(onnx_model_question_paper_Series_Code)
model_Roll1 = rt.InferenceSession(onnx_model_Roll1)
model_Roll2 = rt.InferenceSession(onnx_model_Roll2)

model_sitting = rt.InferenceSession(onnx_model_sitting)
model_stream_horizontal = rt.InferenceSession(onnx_model_stream_horizontal)
model_stream_vertical = rt.InferenceSession(onnx_model_stream_vertical)
model_category1 = rt.InferenceSession(onnx_model_category1)
model_category2=rt.InferenceSession(onnx_model_category2)
model_stream_sub_category=rt.InferenceSession(onnx_model_stream_sub_category)
model_booklet_code=rt.InferenceSession(onnx_model_booklet_code)

model_domicle=rt.InferenceSession(onnx_model_domicle)
model_gender=rt.InferenceSession(onnx_model_gender)
model_disabled=rt.InferenceSession(onnx_model_disabled)
model_subject_group=rt.InferenceSession(onnx_model_subject_group)
model_faculty=rt.InferenceSession(onnx_model_faculty)
model_series=rt.InferenceSession(onnx_model_series)
model_series2=rt.InferenceSession(onnx_model_series2)



    ########################################## global variable ###################################################




global SI_user
stop=0
error_result=0
total_image_length=0
SI=0
roi_x=0
roi_y=0
roi_width=0
roi_height= 0
final_similarity_average=0
db_file=""
table_name=""

###############################globally defined list #############################################
column_name_user11=[]
column_name_ocr1=[]
table_col_name=[]
question_list1=[]


nfeatures_list=[]

barcode_points11=[]
roll_points1=[]
roll_points2=[]
registration_points=[]
exam_date_points=[]

college_code_points=[]
question_points_40=[]
question_points_50=[]
question_points_70=[]
question_points_100=[]
question_points_150=[]
question_points_200=[]

question_points=[]
question_paper_Series_Code_points=[]
stream_horizontal_points=[]
stream_vertical_points=[]
course_points=[]
sitting_points=[]
subject_code_points=[]
category1_points=[]
category2_points=[]
stream_subcategory_points=[]
booklet_code_points=[]
gender_points=[]
domicle_points=[]
disabled_points=[]
subject_group_points=[]
booklet_no_points=[]
series_points=[]
faculty_points=[]
booklet_no_points1=[]
series_pattern2=[]
roll_digit_cordinates=[]
booklet_digit_cordinates=[]
college_code_digit_cordinates=[]
exam_digit_cordinates=[]
reg_digit_cordinates=[]
subject_digit_cordinates=[]

admit_card_available=0
Resize_CheckBox='False'
sift_features11=[]



def column_check(df, col_name):
    predicted_col = []  
    notpredicted_col = []

    for i in range(len(df)):
        col_value = str(df[col_name].iloc[i])
        if pd.notnull(col_value) and "-" not in col_value and "*" not in col_value:
            predicted_col.append(df.iloc[i])
        else:
            notpredicted_col.append(df.iloc[i])
    
    return notpredicted_col, predicted_col

def excel_analysis(df, col_name, output_folder):
    Total_omr_sheets = len(df)
    notpredicted_barcode, predicted_barcode = column_check(df, col_name)
    Correct_prediction = len(predicted_barcode)
    Incorrect_prediction = len(notpredicted_barcode)
    col = list(df.columns)
    
    df_barcode_predicted = pd.DataFrame(predicted_barcode, columns=col)
    df_barcode_notpredicted = pd.DataFrame(notpredicted_barcode, columns=col)

    correct_path = os.path.join(output_folder, f"predicted_{col_name}.xlsx")
    incorrect_path = os.path.join(output_folder, f"notpredicted_{col_name}.xlsx")

    df_barcode_predicted.to_excel(correct_path, index=False)
    df_barcode_notpredicted.to_excel(incorrect_path, index=False)

    return Total_omr_sheets, Correct_prediction, Incorrect_prediction, correct_path, incorrect_path




def generate_demo_answer_sheet(df, question_columns):
     demo_answers = {}
    
     for col in question_columns:
         answer_counts = df[col].value_counts()
         most_selected_answer = answer_counts.idxmax() if not answer_counts.empty else None
         demo_answers[col] = most_selected_answer
    
     return demo_answers


def calculate_difficulty(df):
    difficulty = {}
    
    # Valid answer choices
    valid_answers = ['A', 'B', 'C', 'D']
    
    for question in df.columns:
        if not question.startswith('Q'):
            continue
        
        # Get responses and treat '-' and '*' as NaN (skipped or incorrect responses)
        responses = df[question].replace({'-': np.nan, '*': np.nan})
        
        # Drop NaN values (non-answers)
        valid_responses = responses.dropna()

        # Calculate response frequencies for valid answers
        response_counts = valid_responses.value_counts()
        total_responses = len(df[question])  # Total responses including blanks
        valid_response_count = len(valid_responses)  # Responses without skips
        
        # Calculate the unanswered proportion (how many skipped the question)
        unanswered_count = total_responses - valid_response_count
        unanswered_proportion = unanswered_count / total_responses if total_responses > 0 else 0
        
        # Determine the proportion of students who selected the most frequent answer
        if valid_response_count > 0:
            most_frequent_answer_count = response_counts.max()  # Highest frequency of any valid answer
            consensus_proportion = most_frequent_answer_count / valid_response_count
        else:
            consensus_proportion = 0  # No valid responses means no consensus
        
        # Difficulty score: low consensus + more unanswered responses = harder question
        difficulty_score = (1 - consensus_proportion) + unanswered_proportion
        
        difficulty[question] = difficulty_score
    
    return difficulty


class WorkerThread(QThread):
        finished = Signal(int)
        update_progress=Signal(int)
        db_error=Signal(str)
        omr_remaining_time=Signal(str)
        end_timer=Signal(str)
       

        def __init__(self, script_launcher):
            global  folder_path,output_path_final,template_path,db_file,table_name,cordinates_point_path,correct_table_name,\
                    admit_card_column_name,admit_card_table_name, admit_card_available, error_result
          
            super().__init__()
            self.script_launcher = script_launcher

            correct_table_name=self.script_launcher.table_name 
            #Incorrect_table_name=self.script_launcher.table_name2 
            template_path=self.script_launcher.template_path
            db_file=self.script_launcher.db_path_value
            folder_path=self.script_launcher.folder_path
            output_path_final=self.script_launcher.output_path_final
            admit_card_table_name=self.script_launcher.admit_card_database_table
            #admit_card_column_name= self.script_launcher.admit_card_database_column_name


            if admit_card_table_name!='':
                admit_card_available=1
                
            else:
                admit_card_available=0
                
           
     

    

        def run(self):

            global column_name_user1, SI,final_similarity_average,column_list,question_list_user1,column_name_user,sift_value,nfeatures_list
            global Image_Width, Image_Height,error_result,column_name_user11,sift_features_list,sift_features,sift_features11,sift_features1
            #sift_features1,sift_features_list,sift_features,sift_features11
            global question_list_user1,question_list,question_list1,column_name_admit_card,ocr_column,column_name_ocr1
            
            column_name_user1 = self.script_launcher.columns_platform.text()
            
            column_name_user11.append(column_name_user1)
            column_name_user=column_name_user11
            column_name_user11=[]

            question_list_user1= self.script_launcher.columns_platform_question.text()
            question_list1.append(question_list_user1)
            question_list= question_list1
            question_list_user1=[]
            question_list1=[]
            
            ocr_column = self.script_launcher.columns_platform_ocr.text()
            column_name_ocr1.append(ocr_column)
            column_name_admit_card=column_name_ocr1
            ocr_column =[]
            column_name_ocr1=[]


            sift_features1=self.script_launcher.columns_platform_sift.currentText()
            if sift_features1=='':
                sift_features=0
            else:
                sift_features11.append(sift_features1)
                sift_features_list= sift_features11
                sift_features_org=sift_features_list[0]
                sift_features=int(sift_features_org)
            
                    
                sift_features11=[]


            similarity_per_value1 = self.script_launcher.similarity_per.text()
            img_width = self.script_launcher.image_Width.text()
            img_height = self.script_launcher.image_Height.text()

            SI_user = similarity_per_value1
            SI=int(SI_user)
    


            Image_Width=int(img_width) 
            Image_Height=int(img_height)

         
            self.corrdinates_points_process()
        
            start = time.time()
       
            self.omr_check_function()
            end = time.time()
            total_time=end-start
            #total_time_hours = total_time / 3600
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            total_time_hours=f'{int(hours)} hr {int(minutes)} min {int(seconds)} sec'
            self.end_timer.emit(total_time_hours)
           
         
            self.finished.emit(error_result)
            
      
    
            

#################### function for cor-rdinates points #########################
    

        def corrdinates_points_process(self):
            global roi_x,roi_y,roi_width,roi_height,barcode_points11,\
             roll_points1,roll_points2,exam_date_points,registration_points,\
             college_code_points,question_paper_Series_Code_points,course_points,sitting_points,stream_horizontal_points,stream_vertical_points,\
                   question_points_40,question_points_50,question_points_70,question_points_100,question_points_150,question_points_200,question_points,\
                   subject_code_points,category1_points,category2_points,stream_subcategory_points,booklet_code_points,\
                   gender_points,domicle_points,disabled_points,subject_group_points,booklet_no_points,series_points,\
            faculty_points,column_list,booklet_no_points1,series_pattern2
           
         
          
            first_string = column_name_user[0]
            column_list = [item.strip() for item in first_string.split(',')]
          

            if 'Barcode' in column_list:
                #print(" Bracode Found")
                barcode_points11=barcode_points11_user 
                roi_x, roi_y, roi_width, roi_height = barcode_points11[0] 
                #print(" barcode_points",barcode_points11)
               
            if 'Roll_No_0_9' in column_list:
                    roll_points1=roll_points1_user
                    #print("roll_points11",roll_points1)
               
            if 'Roll_No_1_0' in column_list:
               
                roll_points2=roll_points2_user
           
            if 'Registration' in column_list:
                registration_points=registration_points_user
               
            if 'Exam_Date' in column_list:
                exam_date_points=exam_date_points_user
               
            if 'College_Code' in column_list:
                college_code_points=college_code_points_user
               
            if 'Course' in column_list:
                course_points=course_points_user
              
            if 'Question_paper_Series_Code' in column_list:
                question_paper_Series_Code_points=question_paper_Series_Code_points_user
            
            if 'Sitting' in column_list:
                sitting_points=sitting_points_user
               
            if 'Stream_Horizontal' in column_list:
                stream_horizontal_points=stream_horizontal_points_user
                
            if 'Stream_Vertical' in column_list:     
                stream_vertical_points=stream_vertical_points_user
                
              
            if 'Subject_Code' in column_list:
                subject_code_points=subject_code_points_user 
            
            if 'Category_Pattern_Gen_BC_SC_ST' in column_list:
                category1_points= category1_points_user
              
            if 'Category_Pattern_OBC_UR_SC_ST' in column_list:
                category2_points= category2_points_user
               
            if 'Stream_Others_Vocational_General_Honours' in column_list:
                stream_subcategory_points= stream_subcategory_points_user
                
              
            if 'Booklet_Code_Series' in column_list:
               
                booklet_code_points= booklet_code_points_user
                
            if 'Gender' in column_list:
                gender_points= gender_points_user
               
            if 'Domicile' in column_list:
                domicle_points= domicle_points_user
                
       
            if 'Physically_Disabled' in column_list:
                disabled_points= disabled_points_user
                
            if 'Subject_PCM_PCB' in column_list:
                subject_group_points= subject_group_points_user
                
      
            if 'Booklet_No_0_9' in column_list:
                booklet_no_points1= booklet_no_points1_user
          
            if 'Booklet_No_1_0' in column_list:
                booklet_no_points= booklet_no_points_user
     
            if 'Series' in column_list:
                series_points= series_points_user
             
            if 'Series2'  in column_list:
                series_pattern2= series_pattern2_user
                
            if 'Faculty' in column_list:
                faculty_points= faculty_points_user
               
            if '40' in question_list:
                question_points_40=question_points_40_user
                question_points=question_points_40
               
                

            if '50' in question_list:
                question_points_50=question_points_50_user
                question_points=question_points_50
           


            if '70' in question_list:
                question_points_70=question_points_70_user
                question_points=question_points_70
                
                
            if '100' in question_list:
                question_points_100=question_points_100_user
                question_points=question_points_100

            if '150' in question_list:
                question_points_150=question_points_150_user
                question_points=question_points_150
            if '200' in question_list:
                question_points_200=question_points_200_user
                question_points=question_points_200

             
 



###########################omr check ##############################################################            
        def omr_check_function(self):
           
            question_mapping={11:'A',12:'B', 13:'C',14:'D', 24:'-', 25:'*'}
            roll1_mapping= {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '*', 11:'-'}
            #roll1_mapping={0: '0', 1: '1', 2: '2', 3: '3',4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 11:'-',10:'*'}
            
            roll2_mapping={0: '0', 1: '1', 2: '2', 3: '3',4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',10:'A',11:'B',12:'C',13:'SS',15:'acs_blank',16:'-',17:'*'}
            booklet_no_maping={0: '0', 1: '1', 2: '2', 3: '3',4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',10:'A',11:'B',12:'C',13:'SS',15:'acs_blank',16:'-',17:'*'}
           
            registration_mapping={0: '0', 1: '1', 2: '2', 3: '3',4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 26:'-',27:'*'}
            college_code_mapping={0: '0', 1: '1', 2: '2', 3: '3',4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 26:'-',27:'*'}
            stream_mapping={41:'Arts',42:'Commerce',43:'Science',44:'-',45:'*'}
            course_mapping={17:'Others',18:'Honours',19:'General',20:'Vocational',28:'-', 29:'*'}
            #sitting_mapping={37:'1st_sitting',38:'2nd_sitting',39:'-',40:'*'}
            sitting_mapping={1:'1st_sitting',2:'2nd_sitting',3:'-',4:'*'}
            series_mapping={31:'series_A',32:'series_B',33:'series_C',34:'series_D', 35:'-',36:'*'}
            horizontal_series_mapping={1:'series_A',2:'series_B',3:'series_C',4:'series_D', 5:'-',6:'*'}

            exam_date_mapping={0: '0', 1: '1', 2: '2', 3: '3',4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 26:'-',27:'*'}
            subject_code_mapping={0: '0', 1: '1', 2: '2', 3: '3',4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 26:'-',27:'*'}
            category1_mapping = {1: 'BC-I', 2: 'BC-II', 3: 'General', 4: 'SC', 5: 'ST', 7: '-', 8: '*'}

           
            category2_mapping={11:'OBC',12:'SC',13:'ST',14:'UR'}
            
            stream2_mapping= {11:'General', 12:'Honours', 13:'Others', 14:'Vocational', 15:'-'}

            gender_mapping={23:'Female',24:'Male',25:'-'}
            domicle_disabled_mapping= {21:'NO',22:'Yes'}
            subject_group_mapping={16:'PCB', 17:'PCM', 18:'-', 19:'*'}
            
            faculty_mapping= {41:'Arts',42:'Commerce',43:'Science'}
            
            batch_size = 1
                    
            def prediction_classification(image_path, x, y, width,height,model2):
                image = cv2.imread(image_path)
                cropped_image = image[y:y + height, x:x + width]
                results = model2(cropped_image,verbose=False) 
                name_dict=results[0].names  
                prob1=results[0].probs 
                label_top=prob1.top1 
                confidence=prob1.top1conf.numpy()
                confidence_top=confidence.tolist() 
                if  label_top is None: 
                        roll_label="*"
                elif label_top==10:
                    roll_label="-"
                else:
                    roll_label=str(label_top)
                return roll_label
            
            # def prediction_classification(image_path, x, y, width,height,onnx_session):
            #     # Resize image to the size expected by the model (64x64)
            #     image = cv2.imread(image_path)
            #     image = image[y:y + height, x:x + width]
            #     image_resized = cv2.resize(image, (64, 64))
                
            #     # Preprocess the image (add batch dimension and change shape to NCHW)
            #     image_preprocessed = image_resized.astype(np.float32) / 255.0  # Normalize
            #     image_preprocessed = np.transpose(image_preprocessed, (2, 0, 1))  # Convert to (C, H, W)
            #     image_preprocessed = np.expand_dims(image_preprocessed, axis=0)  # Add batch dimension

            #     # Run inference using ONNX Runtime
            #     input_name = onnx_session.get_inputs()[0].name
            #     outputs = onnx_session.run(None, {input_name: image_preprocessed})

            #     # Process the output (assuming it's structured as [labels, probabilities, etc.])
            #     # You'll need to adapt this part to match the actual output of your model.
            #     label_top = np.argmax(outputs[0])  # Assuming the first output is the probabilities
  
            #     confidence_top = np.max(outputs[0])  # Max confidence score


            #     if label_top == 10:
            #         roll_label = "-"
            #     elif label_top is None:
            #         roll_label = "*"
            #     else:
            #         roll_label = str(label_top)

            #     return roll_label

            def read_barcode_from_roi(image_path, x, y, width, height):

                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                roi = image[y:y + height, x:x + width]
                barcodes = decode(roi)

                barcode_data = ''
                for barcode in barcodes:
                    barcode_data += barcode.data.decode('utf-8')
                return barcode_data

        
            def predict_question_40(image_path, x, y, width, height):
                image =cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(350,35))
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)
                input_name = model_Q_40.get_inputs()[0].name
                output_name = model_Q_40.get_outputs()[0].name
                prediction = model_Q_40.run([output_name], {input_name: image_array})
                predicted_class = np.argmax(prediction)
                predicted_class =  question_mapping.get(predicted_class)  
                predicted_label = str(predicted_class)
                return predicted_label
            
            def predict_question_50(image_path, x, y, width, height):
                image =cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(350,35))
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)
                input_name = model_Q_40.get_inputs()[0].name
                output_name = model_Q_40.get_outputs()[0].name
                prediction = model_Q_40.run([output_name], {input_name: image_array})
                predicted_class = np.argmax(prediction)
                predicted_class =  question_mapping.get(predicted_class)  
                predicted_label = str(predicted_class)
                return predicted_label
            
            def predict_question_70(image_path, x, y, width, height):
                image =cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(350,35))
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)
                input_name = model_Q_40.get_inputs()[0].name
                output_name = model_Q_40.get_outputs()[0].name
                prediction = model_Q_40.run([output_name], {input_name: image_array})
                predicted_class = np.argmax(prediction)
                predicted_class =  question_mapping.get(predicted_class)  
                predicted_label = str(predicted_class)
                return predicted_label
            
            def predict_question_100(image_path, x, y, width, height):
                image =cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(250,50))
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)
                input_name = model_Q_100.get_inputs()[0].name
                output_name = model_Q_100.get_outputs()[0].name
                prediction = model_Q_100.run([output_name], {input_name: image_array})
                predicted_class = np.argmax(prediction)
                predicted_class =  question_mapping.get(predicted_class)  
                predicted_label = str(predicted_class)
                return predicted_label
            
            def predict_question_150(image_path, x, y, width, height):
                image =cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(250,50))
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)
                input_name = model_Q_150.get_inputs()[0].name
                output_name = model_Q_150.get_outputs()[0].name
                prediction = model_Q_150.run([output_name], {input_name: image_array})
                predicted_class = np.argmax(prediction)
                predicted_class =  question_mapping.get(predicted_class)  
                predicted_label = str(predicted_class)
                return predicted_label
            def predict_question_200(image_path, x, y, width, height):
                image =cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(250,50))
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)
                input_name = model_Q_100.get_inputs()[0].name
                output_name = model_Q_100.get_outputs()[0].name
                prediction = model_Q_100.run([output_name], {input_name: image_array})
                predicted_class = np.argmax(prediction)
                predicted_class =  question_mapping.get(predicted_class)  
                predicted_label = str(predicted_class)
                return predicted_label


            def predict_roll_number1(image_path, x, y, width, height):
                image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(40 ,350))
                #resized_image = cv2.resize(cropped_image,(350 ,40))
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
                image_array = np.expand_dims(image_array, axis=-1) 
                input_name = model_Roll1.get_inputs()[0].name
                output_name = model_Roll1.get_outputs()[0].name
                prediction = model_Roll1.run([output_name], {input_name: image_array})[0]
                predicted_class = np.argmax(prediction)
                if predicted_class == 11:
                    predicted_label='-'
                elif predicted_class == 10:
                    predicted_label='*'
                else:
                    predicted_label = roll1_mapping.get(predicted_class, '')
                return predicted_label


            # def predict_roll_number1(image_path,x,y,width,height):
            #     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            #     resized_image = cv2.resize(image, (40,350)) 
            #     image_array = resized_image / 255.0
            #     image_array = image_array.astype(np.float32)
            #     image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
            #     image_array = np.expand_dims(image_array, axis=-1)
            #     # Run inference
            #     input_name = model_Roll1.get_inputs()[0].name
            #     output_name = model_Roll1.get_outputs()[0].name
            #     prediction= model_Roll1.run([output_name], {input_name: image_array})[0]

            #     predicted_class = np.argmax(prediction)
            #     print("prediction category",predicted_class)
            #     # Map predicted class to label
            #     predicted_label = roll1_mapping.get(predicted_class, '')
            #     print("Predicted Label ca:", predicted_label)
            #     return predicted_label
            
            def predict_booklet_no1(image_path, x, y, width, height):
                image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(40 ,350))
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
                image_array = np.expand_dims(image_array, axis=-1) 
                input_name = model_Roll1.get_inputs()[0].name
                output_name = model_Roll1.get_outputs()[0].name
                prediction = model_Roll1.run([output_name], {input_name: image_array})[0]
                predicted_class = np.argmax(prediction)
                if predicted_class == 11:
                    predicted_label='-'
                elif predicted_class == 10:
                    predicted_label='*'
                else:
                    predicted_label = roll1_mapping.get(predicted_class, '')
                return predicted_label
            
            def predict_roll_number2(image_path, x, y, width, height):
                image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(50 ,310))
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
                image_array = np.expand_dims(image_array, axis=-1) 
                input_name = model_Roll2.get_inputs()[0].name
                output_name = model_Roll2.get_outputs()[0].name
                prediction = model_Roll2.run([output_name], {input_name: image_array})[0]
                predicted_class = np.argmax(prediction)
                predicted_label = roll2_mapping.get(predicted_class, '')
                return predicted_label


            def predict_booklet_no2(image_path, x, y, width, height):
                
                image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(50 ,310))
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)

                input_name =  model_Roll2.get_inputs()[0].name
                output_name =  model_Roll2.get_outputs()[0].name
                prediction =  model_Roll2.run([output_name], {input_name: image_array})[0]
                predicted_class = np.argmax(prediction)
                predicted_label = booklet_no_maping.get(predicted_class, '')
                return predicted_label
            
            def predict_registration(image_path, x, y, width, height):
                image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(40 ,350))   
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
                image_array = np.expand_dims(image_array, axis=-1)   
                input_name = model_Roll1.get_inputs()[0].name
                output_name = model_Roll1.get_outputs()[0].name
                prediction = model_Roll1.run([output_name], {input_name: image_array})[0]
                predicted_class = np.argmax(prediction)
                if predicted_class == 11:
                    predicted_label='-'
                elif predicted_class == 10:
                    predicted_label='*'
                else:
                    predicted_label = roll1_mapping.get(predicted_class, '')
                return predicted_label

            
            def predict_college_code(image_path, x, y, width, height):
                image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(40 ,350))  
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
                image_array = np.expand_dims(image_array, axis=-1) 
                input_name = model_Roll1.get_inputs()[0].name
                output_name = model_Roll1.get_outputs()[0].name
                prediction = model_Roll1.run([output_name], {input_name: image_array})[0]
                predicted_class = np.argmax(prediction)
                if predicted_class == 11:
                    predicted_label='-'
                elif predicted_class == 10:
                    predicted_label='*'
                else:
                    predicted_label = roll1_mapping.get(predicted_class, '')
                return predicted_label

            def predict_exam_date(image_path, x, y, width, height):
                image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(40 ,350))    
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
                image_array = np.expand_dims(image_array, axis=-1) 
                input_name = model_Roll1.get_inputs()[0].name
                output_name = model_Roll1.get_outputs()[0].name
                prediction = model_Roll1.run([output_name], {input_name: image_array})[0]
                predicted_class = np.argmax(prediction)
                if predicted_class == 11:
                    predicted_label='-'
                elif predicted_class == 10:
                    predicted_label='*'
                else:
                    predicted_label = roll1_mapping.get(predicted_class, '')
                return predicted_label
            
            def predict_subject_code(image_path, x, y, width, height):
                image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(40 ,350)) 
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
                image_array = np.expand_dims(image_array, axis=-1) 
                input_name = model_Roll1.get_inputs()[0].name
                output_name = model_Roll1.get_outputs()[0].name
                prediction = model_Roll1.run([output_name], {input_name: image_array})[0]
                predicted_class = np.argmax(prediction)
                if predicted_class == 11:
                    predicted_label='-'
                elif predicted_class == 10:
                    predicted_label='*'
                else:
                    predicted_label = roll1_mapping.get(predicted_class, '')
                return predicted_label
            
            def predict_course(image_path, x, y, width, height):
                image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(534 ,50))
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)
                input_name =  model_course.get_inputs()[0].name
                output_name =  model_course.get_outputs()[0].name
                prediction =  model_course.run([output_name], {input_name: image_array})[0]
                predicted_class = np.argmax(prediction)
                predicted_label = course_mapping.get(predicted_class, '')
                return predicted_label


            
            def predict_question_paper_Series_Code(image_path, x, y, width, height):
               
                image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(350 ,150))      
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)
                input_name = model_question_paper_Series_Code.get_inputs()[0].name
                output_name = model_question_paper_Series_Code.get_outputs()[0].name
                prediction = model_question_paper_Series_Code.run([output_name], {input_name: image_array})[0]
                predicted_class = np.argmax(prediction)
                predicted_label = series_mapping.get(predicted_class, '')  
                return predicted_label
             
            def predict_sitting(image_path, x, y, width, height):
                image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(150 ,100))   
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)
                input_name = model_sitting.get_inputs()[0].name
                output_name = model_sitting.get_outputs()[0].name
                prediction = model_sitting.run([output_name], {input_name: image_array})[0]
                predicted_class = np.argmax(prediction)
                predicted_label = sitting_mapping.get(predicted_class, '')  
                return predicted_label

            def predict_stream_horizontal(image_path, x, y, width, height):
                image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(500 ,50))
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)
                input_name = model_stream_horizontal.get_inputs()[0].name
                output_name = model_stream_horizontal.get_outputs()[0].name
                prediction = model_stream_horizontal.run([output_name], {input_name: image_array})[0]
                predicted_class = np.argmax(prediction)
                predicted_label = stream_mapping.get(predicted_class, '')
                return predicted_label
            
            def predict_stream_vertical(image_path, x, y, width, height):
               
                image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(150 ,120))
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)

                input_name = model_stream_vertical.get_inputs()[0].name
                output_name = model_stream_vertical.get_outputs()[0].name
                prediction = model_stream_vertical.run([output_name], {input_name: image_array})[0]
                predicted_class = np.argmax(prediction)
                predicted_label = stream_mapping.get(predicted_class, '') 
                return predicted_label


            
            def predict_category1(image_path, x, y, width, height):
                image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(250,350))
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)
                input_name = model_category1.get_inputs()[0].name
                output_name = model_category1.get_outputs()[0].name
                prediction = model_category1.run([output_name], {input_name: image_array})[0]
                predicted_class = np.argmax(prediction)
                predicted_label = category1_mapping.get(predicted_class, '')
                return predicted_label
            
            def predict_category2(image_path, x, y, width, height):
                image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(190,290))
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)
                input_name = model_category2.get_inputs()[0].name
                output_name = model_category2.get_outputs()[0].name
                prediction = model_category2.run([output_name], {input_name: image_array})[0]
                predicted_class = np.argmax(prediction)
                predicted_label = category2_mapping.get(predicted_class, '')
                return predicted_label

            
            def predict_stream_subcategory(image_path, x, y, width, height):
                image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(150,160))
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)
                input_name = model_stream_sub_category.get_inputs()[0].name
                output_name = model_stream_sub_category.get_outputs()[0].name
                prediction = model_stream_sub_category.run([output_name], {input_name: image_array})[0]
                predicted_class = np.argmax(prediction)
                predicted_label = stream2_mapping.get(predicted_class, '')
                return predicted_label
            
            def predict_booklet_code(image_path, x, y, width, height):
                image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(160,140))
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)
                input_name = model_booklet_code.get_inputs()[0].name
                output_name = model_booklet_code.get_outputs()[0].name
                prediction = model_booklet_code.run([output_name], {input_name: image_array})[0]
                predicted_class = np.argmax(prediction)
                predicted_label = series_mapping.get(predicted_class, '')
                return predicted_label
            
            def predict_gender(image_path, x, y, width, height):
                image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(180,60))
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)
                input_name = model_gender.get_inputs()[0].name
                output_name = model_gender.get_outputs()[0].name
                prediction = model_gender.run([output_name], {input_name: image_array})[0]
                predicted_class = np.argmax(prediction)
                predicted_label = gender_mapping.get(predicted_class, '')
                return predicted_label
            
            def predict_domicle(image_path, x, y, width, height):
                image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(150,50))
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)
                input_name = model_domicle.get_inputs()[0].name
                output_name = model_domicle.get_outputs()[0].name
                prediction = model_domicle.run([output_name], {input_name: image_array})[0]
                predicted_class = np.argmax(prediction)
                predicted_label = domicle_disabled_mapping.get(predicted_class, '')
                return predicted_label
            
            def predict_disabled(image_path, x, y, width, height):
                image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(150,50))
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)

                input_name = model_disabled.get_inputs()[0].name
                output_name =model_disabled.get_outputs()[0].name
                prediction = model_disabled.run([output_name], {input_name: image_array})[0]
                predicted_class = np.argmax(prediction)
                predicted_label = domicle_disabled_mapping.get(predicted_class, '')
                return predicted_label
            
            def predict_subject_group(image_path, x, y, width, height):
                image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(150,50))
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)
                input_name = model_subject_group.get_inputs()[0].name
                output_name = model_subject_group.get_outputs()[0].name
                prediction = model_subject_group.run([output_name], {input_name: image_array})[0]
                predicted_class = np.argmax(prediction)
                predicted_label = subject_group_mapping.get(predicted_class, '')
                return predicted_label
            
            def predict_series(image_path, x, y, width, height):
                image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(150,150))  
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)
                input_name = model_series.get_inputs()[0].name
                output_name = model_series.get_outputs()[0].name
                prediction = model_series.run([output_name], {input_name: image_array})[0]
                predicted_class = np.argmax(prediction)
                predicted_label = series_mapping.get(predicted_class, '')
                return predicted_label
            
            def predict_Series2(image_path, x, y, width, height):
                image =cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(260,50))
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)
                input_name = model_series2.get_inputs()[0].name
                output_name = model_series2.get_outputs()[0].name
                prediction = model_series2.run([output_name], {input_name: image_array})
                predicted_class = np.argmax(prediction)
                predicted_class =  horizontal_series_mapping.get(predicted_class)  
                predicted_label = str(predicted_class)
                return predicted_label

            def predict_faculty(image_path, x, y, width, height):
                image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                cropped_image = image[y:y + height, x:x + width]
                resized_image = cv2.resize(cropped_image,(150,150))
                image_array = resized_image / 255.0
                image_array = image_array.astype(np.float32)
                image_array = np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)
                input_name = model_faculty.get_inputs()[0].name
                output_name = model_faculty.get_outputs()[0].name
                prediction = model_faculty.run([output_name], {input_name: image_array})[0]
                predicted_class = np.argmax(prediction)
                predicted_label = faculty_mapping.get(predicted_class, '')
                return predicted_label
            
            


            def process_image(filename):
                first_string = column_name_user[0]
                column_list11 = [item.strip() for item in first_string.split(',')]
                image_path = os.path.join(folder_path, filename)
                result_row = {'Filename': os.path.basename(filename)}

                if 'Barcode' in column_list11:
                    barcode_data=''
                    barcode_data = read_barcode_from_roi(
                        image_path, roi_x, roi_y, roi_width, roi_height)
                    result_row['Barcode'] = barcode_data
                  
                if 'Roll_No_0_9' in column_list11:
                    roll = '' 
                    for point in roll_points1:
                           
                            x, y, roll_width, roll_height = point
                            roll += predict_roll_number1(image_path, x, y, roll_width, roll_height)
                    result_row['Roll_No_0_9'] = roll
                    del roll    

                if 'Roll_No_1_0' in column_list11:         
                    roll2 = ''
                    for point in roll_points2:
                            x, y, roll2_width, roll2_height = point
                            roll2 += predict_roll_number2(image_path, x, y, roll2_width, roll2_height)
                    result_row['Roll_No_1_0'] = roll2
                    del roll2 
                    
                if 'Exam_Date' in column_list11:     
                    Exam_date = ''
                    for point in exam_date_points:
                            x, y, EXM_width, EXM_height = point
                            Exam_date += predict_exam_date(image_path, x, y, EXM_width, EXM_height)
                    result_row['Exam_Date'] = Exam_date
                    del Exam_date     
                
                if 'Registration' in column_list11:   
                    registration = ''
                    for point in registration_points:
                            x, y, registration_width, registration_height = point
                            registration += predict_registration(image_path, x, y, registration_width, registration_height)
                    result_row['Registration'] = registration
                    del registration 
                
                if 'College_Code' in column_list11: 
                    college_code = ''
                    for point in college_code_points:
                            x, y, college_code_width, college_code_height = point
                            college_code += predict_college_code(image_path, x, y, college_code_width, college_code_height)
                    result_row['College_Code'] = college_code  
                    del college_code 
            
                if 'Course' in column_list11: 
                    course = ''
                    for point in course_points:
                            x, y,  course_width,  course_height = point
                            course += predict_course(image_path, x, y,  course_width, course_height)
                    result_row['Course'] = course
                    del course 
                
                if 'Subject_Code' in column_list11: 
                    subject_code = ''
                    for point in subject_code_points:
                            x, y, subject_code_width, subject_code_height = point
                            subject_code += predict_subject_code(image_path, x, y, subject_code_width, subject_code_height)
                    result_row['Subject_Code'] = subject_code

                if 'Question_paper_Series_Code' in column_list11: 
                    question_paper_Series_Code = ''
                    for point in question_paper_Series_Code_points:
                            x, y,  question_paper_Series_Code_points_width, question_paper_Series_Code_points_height = point
                            question_paper_Series_Code  += predict_question_paper_Series_Code(image_path, x, y,  question_paper_Series_Code_points_width,question_paper_Series_Code_points_height)
                    result_row['Question_paper_Series_Code'] = question_paper_Series_Code  
                    del  question_paper_Series_Code

              
                if 'Sitting'in column_list11:    
                    sitting = ''
                    for point in sitting_points:
                        x, y,  sitting_width,  sitting_height = point
                        sitting+= predict_sitting(image_path, x, y,  sitting_width, sitting_height)
                    result_row['Sitting'] = sitting
                    del sitting
                 
                if 'Stream_Horizontal'in column_list11:  
                    stream_horizontal = ''
                    for point in stream_horizontal_points:
            
                            x, y,  stream_horizontal_width,  stream_horizontal_height = point
                            stream_horizontal += predict_stream_horizontal(image_path, x, y,  stream_horizontal_width, stream_horizontal_height)
                    result_row['Stream_Horizontal'] = stream_horizontal
                    del stream_horizontal
        
                if 'Stream_Vertical' in column_list11:  
                    stream_vertical = ''
                    for point in stream_vertical_points:
                            x, y,  stream_vertical_width,  stream_vertical_height = point
                            stream_vertical  += predict_stream_vertical(image_path, x, y,  stream_vertical_width, stream_vertical_height)
                    result_row['Stream_Vertical'] = stream_vertical     
                    del stream_vertical 
                
                if 'Category_Pattern_Gen_BC_SC_ST' in column_list11: 
                    category1 = ''
                    for point in category1_points :
                            x, y, category1_width,  category1_height = point
                            category1 += predict_category1(image_path, x, y,  category1_width, category1_height)
                    result_row['Category_Pattern_Gen_BC_SC_ST'] = category1
                    del category1
                
                if 'Category_Pattern_OBC_UR_SC_ST'in column_list11: 
                    category2 = ''
                    for point in category2_points :
                            x, y, category2_width,  category2_height = point
                            category2 += predict_category2(image_path, x, y,  category2_width, category2_height)
                    result_row['Category_Pattern_OBC_UR_SC_ST'] = category2
                    del category2

                if 'Stream_Others_Vocational_General_Honours' in column_list11: 
                    stream_subcategory=''
                    for point in stream_subcategory_points :
                            x, y, stream2_width, stream2_height = point
                            stream_subcategory += predict_stream_subcategory(image_path, x, y,  stream2_width, stream2_height)
                    result_row['Stream_Others_Vocational_General_Honours'] =  stream_subcategory

                    del  stream_subcategory
                
                if 'Booklet_Code_Series' in column_list11: 
                    booklet_code=''
                    for point in booklet_code_points :
                            x, y, booklet_code_width, booklet_code_height = point
                            booklet_code += predict_booklet_code(image_path, x, y,  booklet_code_width, booklet_code_height)
                    result_row['Booklet_Code_Series'] = booklet_code
                    del  booklet_code

                if 'Gender' in column_list11: 
                    gender=''
                    for point in gender_points :
                            x, y, gender_width, gender_height = point
                            gender += predict_gender(image_path, x, y,  gender_width, gender_height)
                    result_row['Gender'] = gender
                    del  gender

                if 'Domicile' in column_list11: 
                    domicle=''
                    for point in domicle_points :
                            x, y, domicle_width, domicle_height = point
                            domicle += predict_domicle(image_path, x, y,  domicle_width, domicle_height)
                    result_row['Domicile'] = domicle
                    del  domicle

                if 'Physically_Disabled' in column_list11: 
                    disabled=''
                    for point in disabled_points :
                            x, y, disabled_width, disabled_height = point
                            disabled += predict_disabled(image_path, x, y,  disabled_width, disabled_height)
                    result_row['Physically_Disabled'] = disabled
                    del  disabled
                
                if 'Subject_PCM_PCB' in column_list11: 
                    subject_group=''
                    for point in subject_group_points :
                            x, y, subject_group_width, subject_group_height = point
                            subject_group += predict_subject_group(image_path, x, y,  subject_group_width, subject_group_height)
                    result_row['Subject_PCM_PCB'] = subject_group
                    del subject_group
                
                if 'Booklet_No_0_9' in column_list11: 
                    booklet_no1=''
                    for point in booklet_no_points1 :
                            
                            x, y, booklet_no1_width, booklet_no1_height = point
                            booklet_no1 += predict_booklet_no1(image_path, x, y, booklet_no1_width, booklet_no1_height)
                    result_row['Booklet_No_0_9'] = booklet_no1
                    del booklet_no1
                
                if 'Booklet_No_1_0' in column_list11: 
                    booklet_no=''
                    for point in booklet_no_points :
                      
                            x, y, booklet_no_width, booklet_no_height = point
                            booklet_no += predict_booklet_no2(image_path, x, y,  booklet_no_width, booklet_no_height)
                    result_row['Booklet_No_1_0'] = booklet_no
                    del booklet_no
                
                if 'Series' in column_list:
                    series=''
                    for point in series_points :
                    
                            x, y, series_width, series_height = point
                            series += predict_series(image_path, x, y,  series_width, series_height)
                    result_row['Series'] = series
                    del series

                if 'Series2' in column_list:
                    series2=''
                    for point in series_pattern2 :
                       
                            x, y, series_width, series_height = point
                            series2 += predict_Series2(image_path, x, y,  series_width, series_height)
                    result_row['Series2'] = series2
                    del series2
                
                if 'Faculty' in column_list:
                    faculty=''
                    for point in faculty_points :
                    
                            x, y, faculty_width, faculty_height = point
                            faculty += predict_faculty(image_path, x, y, faculty_width, faculty_height)
                    result_row['Faculty'] = faculty
                    del faculty


                if question_points is not None:
                    for i, point in enumerate(question_points):
                     
                        x, y, question_width, question_height = point
                        if '40' in question_list:
                            label = predict_question_40(image_path, x, y, question_width, question_height)
                            result_row[f'Q{i + 1}'] = label
                        if '50' in question_list:
                            label = predict_question_50(image_path, x, y, question_width, question_height)
                            result_row[f'Q{i + 1}'] = label
                            #ab.append(label)
                       
                        if '70' in question_list:
                         
                            label = predict_question_70(image_path, x, y, question_width, question_height)
                            result_row[f'Q{i + 1}'] = label
                        if '100' in question_list:
                            label = predict_question_100(image_path, x, y, question_width, question_height)
                            result_row[f'Q{i + 1}'] = label

                        if '150' in question_list:
                            label = predict_question_150(image_path, x, y, question_width, question_height)
                            result_row[f'Q{i + 1}'] = label
                        if '200' in question_list:
                            label = predict_question_200(image_path, x, y, question_width, question_height)
                            result_row[f'Q{i + 1}'] = label

                #del image_path, barcode_data, roll_no, Exam_date,  Sitting, x, y, question_width, question_height, label
                
                return result_row


            template_image = cv2.imread(template_path)
            template_image=cv2.resize(template_image,(Image_Width, Image_Height))
            
            
            def straighten_image(input_path, template_image, Image_Width, Image_Height):
                # Load input image
                print("straighten_image input_path",input_path)
  
                image = cv2.imread(input_path)
                if image is None:
                    print(f"Error: Unable to read the image from {input_path}.")
                    return None
                
                # Convert to grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Create SIFT detector with a reduced number of features
                sift = cv2.SIFT_create(nfeatures=sift_features)

                # Detect keypoints and descriptors
                keypoints_template, descriptors_template = sift.detectAndCompute(template_image, None)
                keypoints_image, descriptors_image = sift.detectAndCompute(gray_image, None)

                # Use FLANN-based matcher for faster matching
                index_params = dict(algorithm=1, trees=5)
                search_params = dict(checks=50)  # Fewer checks to speed up matching
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(descriptors_template, descriptors_image, k=2)

                # Apply ratio test to filter good matches
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:  # 0.7 is a common ratio for filtering
                        good_matches.append(m)

                # Check if enough matches are found
                if len(good_matches) < 4:
                    print("Error: Not enough matches found.")
                    return None

                # Extract matched points
                points_template = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                points_image = np.float32([keypoints_image[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Calculate the homography matrix with a relaxed threshold for RANSAC
                transformation_matrix, _ = cv2.findHomography(points_image, points_template, cv2.RANSAC, 3.0)  # Increased threshold

                # Apply the perspective warp
                height, width = image.shape[:2]
                straightened_image = cv2.warpPerspective(image, transformation_matrix, (width, height))

                # Create a mask for valid pixels
                mask = np.where((straightened_image[:, :, 0] != 0) | (straightened_image[:, :, 1] != 0) | (straightened_image[:, :, 2] != 0), 255, 0).astype(np.uint8)
                straightened_image = cv2.bitwise_and(straightened_image, straightened_image, mask=mask)

                # Resize to desired dimensions
                straightened_image = cv2.resize(straightened_image, (Image_Width, Image_Height))
                
                return straightened_image


            def straighten_images_with_sift(image_paths, output_folder, template_image,SI,count,batch_count,
                                            image_batches,final_similarity_average,Image_Width, Image_Height):
                straightened_image_paths = []
                similarity_index=0
               
                os.makedirs(output_folder, exist_ok=True)
                unmatched_folder = os.path.join(output_folder, "unmatched")
                os.makedirs(unmatched_folder, exist_ok=True)
                straightened_folder = os.path.join(output_folder, "straightened")
                os.makedirs(straightened_folder, exist_ok=True)


                similarity_index = SI
                
                def process_image1(input_path):
                    straightened_image = straighten_image(input_path, template_image,Image_Width, Image_Height)
             
                    if straightened_image is not None:
                        base_filename = os.path.basename(input_path)

                        similarity = cv2.matchTemplate(template_image, straightened_image, cv2.TM_CCOEFF_NORMED)
                        similarity_percentage = similarity.max() * 100
                        print("similarity_percentage",similarity_percentage)
                        #print("similarity_percentage",similarity_percentage)
                        if similarity_percentage >=similarity_index:
                           
                            
                            #gray_image = cv2.cvtColor(straightened_image, cv2.COLOR_BGR2GRAY)

                            straightened_path = os.path.join(straightened_folder, base_filename)
                            cv2.imwrite(straightened_path,straightened_image)

                            straightened_image_paths.append(straightened_path)

                            self.update_progress.emit(batch_count)
                        else:
                            # Save the unmatched image to the unmatched folder
                            unmatched_path = os.path.join(unmatched_folder, base_filename)
                            cv2.imwrite(unmatched_path, cv2.imread(input_path))
                #process_image1(image_paths)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    executor.map(process_image1, image_paths)

                return straightened_image_paths

           

            import concurrent.futures

            def is_scientific_notation(number_str):
                # Regular expression to match scientific notation
                scientific_notation_pattern = re.compile(r'^[+-]?\d+(\.\d+)?[eE][+-]?\d+$')
                return bool(scientific_notation_pattern.match(number_str))
            
            def count_value(count):
                 count+=1
                 return count
            
            def open_connection_db_admit_card(db_path):
                try:
                    conn_admit_card = sqlite3.connect(db_path)
                    
                    return conn_admit_card 
                except sqlite3.Error as e:
                    print(f"SQLite error: {e}")
                    return None
            def close_connection_db_admit_card(conn):
                if conn:
                    conn.close()
                    print("Connection closed.") 
                        
          
            def database_omr_table(db_file,table_name,conn):
                
                global error_result
                try:
                    cursor = None
                    cursor = conn.cursor()

                    # Construct the column list for the database table
                    first_string_DB = column_name_user[0]
                    column_list_DB = [item.strip() for item in first_string_DB.split(',')]
                    columns = ['Filename'] + column_list_DB + [f'Q{i}' for i in range(1, len(question_points) + 1)]

                    # Create the table if it does not exist
                    column_definitions = [f'{col} TEXT' for col in columns]
                    create_table_sql = f'''
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            {', '.join(column_definitions)}
                        )
                    '''
                    with conn:
                        conn.execute(create_table_sql)

                    conn.commit()   
                except Exception as e1:
                    error_result=1
                    db_connection_error=f"{e1}"
                    self.db_error.emit(db_connection_error)
                    #self.close()    
               
    
            def database_ocr(db_file,table_name,conn):
                
                global error_result
                try:
                    cursor = None
                    cursor = conn.cursor()

                    # Construct the column list for the database table
                    first_string_DB = column_name_user[0]
                    column_list_DB = [item.strip() for item in first_string_DB.split(',')]
                    first_string =  column_name_admit_card[0]
                    ocr_list11 = [item.strip() for item in first_string.split(',')]
                    
                    ocr_column = [f'{x}_Prediction_Type' for x in ocr_list11]
                   
                    columns = ['Filename'] + column_list_DB+ocr_column + [f'Q{i}' for i in range(1, len(question_points) + 1)]

                    # Create the table if it does not exist
                    column_definitions = [f'{col} TEXT' for col in columns]
                    create_table_sql = f'''
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            {', '.join(column_definitions)}
                        )
                    '''
                    with conn:
                        conn.execute(create_table_sql)

                    conn.commit()   
                except Exception as e1:
                    error_result=1
                    db_connection_error=f"{e1}"
                    self.db_error.emit(db_connection_error)
                    #self.close()    


            def database_omr_insert_table(db_file,table_name,result_row,conn):
             
                global error_result
                try:
                    cursor = None
                    cursor = conn.cursor()

                    first_string_DB = column_name_user[0]
                    column_list_DB = [item.strip() for item in first_string_DB.split(',')]
                    columns = ['Filename'] + column_list_DB + [f'Q{i}' for i in range(1, len(question_points) + 1)]

                    # Prepare the insert statement
                    placeholders = ', '.join(['?'] * len(columns))
                    insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
                    values = [result_row.get(column, '') for column in columns]

                    # Execute the insert statement
                    cursor.execute(insert_sql, values)
                    conn.commit()
                    
                except Exception as e2:
                    error_result=1 
                    table_connection_error=f"{e2}"
                    self.db_error.emit(table_connection_error)
                    #self.close()
            def exit_program():
                sys.exit(0) 
            

            def insert_table_ocr(db_file,table_name,result_row,conn):
             
                global error_result
                try:
                    cursor = None
                    cursor = conn.cursor()

                    first_string_DB = column_name_user[0]
                    column_list_DB = [item.strip() for item in first_string_DB.split(',')]
                    first_string =  column_name_admit_card[0]
                    ocr_list11 = [item.strip() for item in first_string.split(',')]
                   
                    ocr_column = [f'{x}_Prediction_Type' for x in ocr_list11]
                    columns = ['Filename'] + column_list_DB + ocr_column+[f'Q{i}' for i in range(1, len(question_points) + 1)]

                    # Prepare the insert statement
                    placeholders = ', '.join(['?'] * len(columns))
                    insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
                    values = [result_row.get(column, '') for column in columns]

                    # Execute the insert statement
                    cursor.execute(insert_sql, values)
                    conn.commit()
                    
                except Exception as e2:
                    error_result=1 
                    table_connection_error=f"{e2}"
                    self.db_error.emit(table_connection_error)
                    #self.close()
            def exit_program():
                sys.exit(0) 


  
            
            def check_roll_in_database(conn, table_name, column_name, roll_no):
                cursor = None
                try:
                    cursor = conn.cursor()  # Create a cursor from the open connection

                    # Build SQL query using proper string formatting for table and column names
                    query = f"SELECT * FROM {table_name} WHERE {column_name} = ?"
                    cursor.execute(query, (roll_no,))  # Parameterized query to prevent SQL injection

                    existing_user = cursor.fetchone()

                    # Check if the roll number exists
                    if existing_user:
                        return 1
                    else:
                        return 0

                except sqlite3.Error as e:
                
                    return 0, str(e)

                finally:
                    if cursor:
                        cursor.close()
            
           
            def call_digit_model(straightened_image,model2,col_value_orginal,digit_cordinates):
               
                index = [i for i, letter in enumerate(col_value_orginal) if letter == '-' or letter == '*']
                col_value_original_list=list(col_value_orginal)# converting omr roll number into list
                for i in index:
                    points=digit_cordinates[i]
                    x, y, width,height = points
                    roll_classification = prediction_classification(straightened_image, x, y,width,height,model2)
                    col_value_original_list[i]=roll_classification 
                # print("roll_classification",roll_classification)
                
                col_value_original_list_str = ''.join(col_value_original_list)### converting omr roll number after classification in string
                col_value_original_list_str_length=len(col_value_original_list_str)
                
                return col_value_original_list_str  ,col_value_original_list_str_length
            
            def find_damage_col(col_value,col_value_length,majority_length):
                if pd.notnull(col_value) and "-" not in col_value  and "*" not in col_value  and col_value_length==majority_length :
                    #database_omr_insert_table(db_file,correct_table_name,result_row,conn_admit_card)
                    return 0 # no damage    
                else:
                    return 1 # damage 
                
            def process_damage(straightened_image,col_value_orginal,ocr_column_name,dublicate,result_row,ocr_row ,
                               digit_cordinate,Prediction_Type,majority_length ):
                col_value_str ,col_value_str_length= call_digit_model(straightened_image,model2,col_value_orginal,digit_cordinate) 
                if pd.notnull(col_value_str ) and "-" not in col_value_str  and "*" not in col_value_str  and col_value_str_length==majority_length:
                    if col_value_str not in dublicate:### check duplicate(dublicate not found)                    
                        dublicate.append(col_value_str)                                      
                        roll_return_admit_card_db=check_roll_in_database(conn_admit_card,admit_card_table_name,ocr_column_name,col_value_str)
                        if roll_return_admit_card_db==1:# roll found in admit card table
                            result_row[ocr_column_name]=col_value_str
                            ocr_row[Prediction_Type]="ocr"
                        else: # roll not found in admit card table
                            
                            ocr_row[ocr_column_name]=col_value_orginal
                            ocr_row[Prediction_Type]="original"
                    else:# dublicate found
                        
                        ocr_row[ocr_column_name]=col_value_orginal
                        ocr_row[Prediction_Type]="original"
                    
                else: # *, _ and length of digit prediction found
                    ocr_row[ocr_column_name]=col_value_orginal
                    ocr_row[Prediction_Type]="original"

       

            def process_ocr(straightened_image,result_row,column_name_admit_card):
                first_string =  column_name_admit_card[0]
                ocr_list11 = [item.strip() for item in first_string.split(',')]
                
                file_name = result_row["Filename"]
                ocr_row = {'Filename': os.path.basename(file_name)}

                if "Booklet_No_0_9" in ocr_list11:
                   
                    global booklet_dublicate
                    majority_length=len(booklet_digit_cordinates)
                    booklet_dublicate=[]
                    ocr_column_name="Booklet_No_0_9" 
                    Prediction_Type=  "Booklet_No_0_9_Prediction_Type"                
                    col_value_orginal = result_row[ocr_column_name].strip()  # Strip whitespace
                    col_value_length=len(col_value_orginal)   
                    return_value=find_damage_col(col_value_orginal,col_value_length,majority_length)
                    if return_value==0:
                        booklet_dublicate.append(col_value_orginal)
                        ocr_row[Prediction_Type]="original"
                    else:
                        process_damage(straightened_image,col_value_orginal,ocr_column_name,booklet_dublicate,result_row,ocr_row,
                                       booklet_digit_cordinates,Prediction_Type,majority_length )
                        
                if "Booklet_No_1_0" in ocr_list11:
                  
                    global booklet2_dublicate
                    majority_length=len(booklet_digit_cordinates)
                    booklet2_dublicate=[]
                    ocr_column_name="Booklet_No_1_0"   
                    Prediction_Type="Booklet_No_1_0_Prediction_Type"                
                    col_value_orginal = result_row[ocr_column_name].strip()  # Strip whitespace
                    col_value_length=len(col_value_orginal)   
                    return_value=find_damage_col(col_value_orginal,col_value_length,majority_length)
                    if return_value==0:
                        booklet2_dublicate.append(col_value_orginal)
                        ocr_row[Prediction_Type]="original"
                    else:
                        process_damage(straightened_image,col_value_orginal,ocr_column_name,booklet2_dublicate,result_row,ocr_row,
                                       booklet_digit_cordinates,Prediction_Type,majority_length )
                         

                if "College_Code" in ocr_list11:
                   
                    global college_dublicate
                    majority_length=len(college_code_digit_cordinates)
                    college_dublicate=[]
                    ocr_column_name="College_Code"  
                    Prediction_Type= "College_Code_Prediction_Type"                  
                    col_value_orginal = result_row[ocr_column_name].strip()  # Strip whitespace
                    col_value_length=len(col_value_orginal)   
                    return_value=find_damage_col(col_value_orginal,col_value_length,majority_length)
                    if return_value==0:
                        college_dublicate.append(col_value_orginal)
                        ocr_row[Prediction_Type]="original"
                    else:
                        process_damage(straightened_image,col_value_orginal,ocr_column_name,college_dublicate,
                                       result_row,ocr_row,college_code_digit_cordinates,Prediction_Type,majority_length )
                         

                if "Exam_Date" in ocr_list11:
                   
                    global exam_dublicate
                    majority_length=len(exam_digit_cordinates)
                    exam_dublicate=[]
                    ocr_column_name="Exam_Date" 
                    Prediction_Type="Exam_Date_Prediction_Type"                   
                    col_value_orginal = result_row[ocr_column_name].strip()  # Strip whitespace
                    col_value_length=len(col_value_orginal)   
                    return_value=find_damage_col(col_value_orginal,col_value_length,majority_length)
                    if return_value==0:
                        exam_dublicate.append(col_value_orginal)
                        ocr_row[Prediction_Type]="original"
                    else:
                        process_damage(straightened_image,col_value_orginal,ocr_column_name,exam_dublicate,
                                       result_row,ocr_row,exam_digit_cordinates,Prediction_Type,majority_length )

                if "Registration" in ocr_list11:
                   
                    global Registration_dublicate
                    majority_length=len(reg_digit_cordinates)
                    Registration_dublicate=[]
                    ocr_column_name="Registration" 
                    Prediction_Type="Registration_Prediction_Type"                
                    col_value_orginal = result_row[ocr_column_name].strip()  # Strip whitespace
                    col_value_length=len(col_value_orginal)   
                    return_value=find_damage_col(col_value_orginal,col_value_length,majority_length)
                    if return_value==0:
                        Registration_dublicate.append(col_value_orginal)
                        ocr_row[Prediction_Type]="original"
                    else:
                        process_damage(straightened_image,col_value_orginal,ocr_column_name,Registration_dublicate,
                                       result_row,ocr_row,reg_digit_cordinates,Prediction_Type,majority_length )

                if "Roll_No_0_9" in ocr_list11:
                   
                    
                    global Roll1_dublicate
                    majority_length=len(roll_digit_cordinates)
                    Roll1_dublicate=[]
                    ocr_column_name="Roll_No_0_9"   
                    Prediction_Type= "Roll_No_0_9_Prediction_Type"               
                    col_value_orginal = result_row[ocr_column_name].strip()  # Strip whitespace
                    col_value_length=len(col_value_orginal)   
                    return_value=find_damage_col(col_value_orginal,col_value_length,majority_length)
                    if return_value==0:
                        Roll1_dublicate.append(col_value_orginal)
                        ocr_row[Prediction_Type]="original"
                    else:
                        process_damage(straightened_image,col_value_orginal,ocr_column_name,Roll1_dublicate,
                                       result_row,ocr_row,roll_digit_cordinates,Prediction_Type,majority_length )



                if "Roll_No_1_0" in ocr_list11:
                   
                    global Roll2_dublicate
                    majority_length=len(roll_digit_cordinates)
                    Roll2_dublicate=[]
                    ocr_column_name="Roll_No_1_0" 
                    Prediction_Type= "Roll_No_1_0_Prediction_Type"                  
                    col_value_orginal = result_row[ocr_column_name].strip()  # Strip whitespace
                    col_value_length=len(col_value_orginal)   
                    return_value=find_damage_col(col_value_orginal,col_value_length,majority_length)
                    if return_value==0:
                        Roll2_dublicate.append(col_value_orginal)
                        ocr_row[Prediction_Type]="original"
                    else:
                        process_damage(straightened_image,col_value_orginal,ocr_column_name,Roll2_dublicate,
                                       result_row,ocr_row,roll_digit_cordinates,Prediction_Type,majority_length  )

                if "Subject_Code" in ocr_list11:
                   
                    global subject_dublicate
                    majority_length=len(subject_digit_cordinates)
                    subject_dublicate=[]
                    ocr_column_name="Subject_Code" 
                    Prediction_Type="Subject_Code_Prediction_Type"                 
                    col_value_orginal = result_row[ocr_column_name].strip()  # Strip whitespace
                    col_value_length=len(col_value_orginal)   
                    return_value=find_damage_col(col_value_orginal,col_value_length,majority_length)
                    if return_value==0:
                        subject_dublicate.append(col_value_orginal)
                        ocr_row[Prediction_Type]="original"
                    else:
                        process_damage(straightened_image,col_value_orginal,ocr_column_name,subject_dublicate,
                                       result_row,ocr_row,subject_digit_cordinates,Prediction_Type,majority_length )
                
                return result_row,ocr_row


   #admit_card_column_name,admit_card_table_name
            def process_images_with_batches(folder_path, batch_size,db_file, template_image, SI,output_path_final,final_similarity_average,
                                              Image_Width, Image_Height,correct_table_name):
                global total_image_length,count,start_time,majority_length,conn_admit_card,admit_card_available
                output_folder=output_path_final
               
                count = 0
                start_time=0.0
                image_filenames = [filename for filename in os.listdir(folder_path) if filename.lower().endswith((".jpg", ".jpeg", ".png"))]
                image_batches = [image_filenames[i:i + batch_size] for i in range(0, len(image_filenames), batch_size)]
                total_image_length = len(image_batches)
                #majority_length=len(roll_digit_cordinates)
                conn_admit_card =open_connection_db_admit_card(db_file)

                database_omr_table(db_file,correct_table_name,conn_admit_card)
                if admit_card_available==1:
                    database_ocr(db_file, f'{correct_table_name}_Rechecked_Roll_NO',conn_admit_card)
            
                try:

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        for batch_count, batch_filenames in enumerate(image_batches, start=1):
                            if stop == 1:
                                QMessageBox.showwarning("Warning", "Close button is pressed. Terminating batch processing.")
                                break
                            
                            total_image_left= total_image_length-batch_count
                            if count==10:
                                end_time = time.time()
                                total_time_taken=end_time-start_time
   
                                time_taken_one_image=total_time_taken/10 ##### one image
                                #self.timer_one_image.emit(time_taken_one_image)
                                #total_Processing_time_all_image=time_taken_one_image*(total_image_length-10)
                                total_Processing_time_all_image=time_taken_one_image*(total_image_left)
                           
                            if count>10:
                                total_Processing_time_all_image=total_Processing_time_all_image-time_taken_one_image
                                remaining_time=total_Processing_time_all_image
                                hours, remainder = divmod(remaining_time, 3600)
                                minutes, seconds = divmod(remainder, 60)
                                if is_scientific_notation(str(remaining_time)):

                                    time_left=f'0 hr 0 min 0 sec'
                                    self.omr_remaining_time.emit(time_left)
                                else:
                                    time_left=f'{int(hours)} hr {int(minutes)} min {int(seconds)} sec'
                                    self.omr_remaining_time.emit(time_left)
                              
                                if count==500:
                                    count=0
                                    #print("count value is zero",count)
                                    time_left=f'wait.......'
                                    self.omr_remaining_time.emit(time_left)
                          
                            
                            if count==0:
                                start_time=time.time()
                           
                            count=count_value(count)
                            
                   
                 
                       
                            if Resize_CheckBox == 'True':
                                resized_folder = os.path.join(output_folder, "Resized_Folder")
                                os.makedirs(resized_folder, exist_ok=True)

                                for batch_count11, filename in enumerate(batch_filenames, start=1):
                                    image_path = os.path.join(folder_path, filename)  # Loop through each file in batch_filenames
                                  
                                    
                                    img = cv2.imread(image_path)  # Read the image
                                    if img is None:
                                        print(f"Warning: Unable to read image {image_path}")
                                        continue
                                    
                                    # Use the size of the first image for resizing all images
                                    if batch_count == 1:
                                        first_image_size = (img.shape[1], img.shape[0])

                                   
                                    # Resize the image to the first image's size
                                    resize_path = os.path.join(resized_folder, filename)  # Save resized image with the same filename
                                    resized_img = cv2.resize(img, first_image_size)
                                    cv2.imwrite(resize_path, resized_img)

                                    # Now straighten the images
                                    batch_straightened_images = straighten_images_with_sift(
                                        [os.path.join(resized_folder, filename) for filename in batch_filenames],
                                        output_folder, template_image, SI, count, batch_count, image_batches, final_similarity_average, Image_Width, Image_Height
                                    )
                            else:  
                              
                                batch_straightened_images = straighten_images_with_sift(
                                [os.path.join(folder_path, filename) for filename in batch_filenames],
                                output_folder, template_image, SI, count, batch_count, image_batches,final_similarity_average,Image_Width, Image_Height
                         
                            )
                        
                          
                            batch_results = []
           
                            for straightened_image in batch_straightened_images:
                            
                                result_row = process_image(straightened_image) 
                                batch_results.append(result_row)

                            for result_row in batch_results:
                                file_name = result_row["Filename"]  
                                if admit_card_available==1:# admit card given
                                    result_row_response,ocr_row_response=process_ocr(straightened_image,result_row,column_name_admit_card)      
                                    database_omr_insert_table(db_file,correct_table_name,result_row_response,conn_admit_card)    
                                    insert_table_ocr(db_file,f'{correct_table_name}_Rechecked_Roll_NO',ocr_row_response,conn_admit_card)          
                                else:#### admit card not given by user
                                    print("admit card not given")
                                    database_omr_insert_table(db_file,correct_table_name,result_row,conn_admit_card)  
                            del batch_straightened_images
                            

                except Exception as e:
                    print(f"Error during batch processing: {e}")
                close_connection_db_admit_card(conn_admit_card)   

           
###############################################################################################################################

                   
               
            process_images_with_batches(folder_path, batch_size,db_file, template_image, SI,output_path_final,final_similarity_average,
                                              Image_Width, Image_Height,correct_table_name
                                               )
            
################################# OMR Analyzer code ########################################

@app.route('/')
def upload_file():
    return render_template('input_page.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        logging.warning("No file part in the request")
        return render_template('input_page.html', message="No file part")
    
    file = request.files['file']
    if file.filename == '':
        logging.warning("No selected file")
        return render_template('input_page.html', message="No selected file")
    
    allowed_extensions = {'.xlsx', '.csv', '.xls'}
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        logging.error("Invalid file type uploaded")
        return render_template('input_page.html', message="Invalid file type. Please upload a valid sheet.")

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    try:
        file.save(filepath)
        logging.info(f"File saved successfully: {filepath}")
    except Exception as e:
        logging.error(f"Error saving file: {str(e)}")
        return render_template('input_page.html', message=f"Error saving file: {str(e)}")

    try:
        df = pd.read_excel(filepath)
        logging.info(f"File read successfully: {filepath}")
    except Exception as e:
        logging.error(f"Error reading file: {str(e)}")
        return render_template('input_page.html', message=f"Error reading file: {str(e)}")

    # Store the dataframe in the session
    session['data'] = df.to_dict()  # Store as dictionary
    session['filepath'] = filepath
    
    # Store the selected number of questions in the session
    session['pattern'] = request.form.get('pattern')

    return redirect(url_for('analyze'))



@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    # Check for session data
    if 'data' not in session:
        return redirect(url_for('upload_file'))

    # Load the DataFrame once
    df = pd.DataFrame(session['data'])

    # Get limit and pattern from request or session
    if request.method == 'POST':
        limit = int(request.form.get('limit', 10))  # Updated limit from form
        question_pattern = int(request.form.get('pattern', 50))  # Number of questions
    else:
        limit = 3  # Default limit
        question_pattern = int(session.get('pattern', 50))  # From session

    # Construct expected question columns
    expected_columns = [f'Q{i}' for i in range(1, question_pattern + 1)]
    missing_columns = [col for col in expected_columns if col not in df.columns]

    if missing_columns:
        logging.error(f"Missing columns: {missing_columns}")
        return render_template('input_page.html', message="Some question columns are missing in the sheet: " + ", ".join(missing_columns))

    # Calculate metrics
    total_sheets = len(df['Roll_No_0_9'])
    df['Barcode'] = df['Barcode'].astype(str)
    correct_barcodes = df['Barcode'].apply(lambda x: '-' not in x and '*' not in x and pd.notna(x)).sum()
    null_barcodes_sum = df['Barcode'].isna().sum()
    incorrect_barcodes = df['Barcode'].apply(lambda x: pd.isna(x) or '-' in x or '*' in x).sum()
    # Identify null barcodes
    null_barcodes = df[df['Barcode'].isna()]

    # Ensure output folder exists
    output_folder = os.path.join(app.config['UPLOAD_FOLDER'], "Excel_folder")
    os.makedirs(output_folder, exist_ok=True)

    # Save null barcodes to a separate Excel file for download
    null_barcodes_filename = os.path.join(output_folder, 'null_barcodes.xlsx')
    null_barcodes.to_excel(null_barcodes_filename, index=False)



    # Ensure output folder exists
    output_folder = os.path.join(app.config['UPLOAD_FOLDER'], "Excel_folder")
    os.makedirs(output_folder, exist_ok=True)

    # Excel analysis
    col_name = 'Roll_No_0_9'
    null_roll_nos = df[col_name].isna().sum()
    total_omr_sheets, correct_roll_nos, incorrect_prediction, correct_excel, incorrect_excel = excel_analysis(df, col_name, output_folder)
    incorrect_roll_nos = incorrect_prediction - null_roll_nos

    # Get incorrect roll numbers DataFrame
    incorrect_roll_df = df[df[col_name].isna() | (df[col_name] == 'incorrect')]

    # # Check if the DataFrame is empty
    # if incorrect_roll_df.empty:
    #     print("No incorrect roll numbers found.")
    # else:
    #     print(f"Incorrect roll numbers found: {incorrect_roll_df}")


    # Identify incorrect roll numbers based on presence of '-' or '*'
    incorrect_roll_df = df[df[col_name].str.contains('-|\\*', na=False)]

    # Check if the DataFrame is empty
    # if incorrect_roll_df.empty:
    #     print("No incorrect roll numbers found.")
    # else:
    #     print(f"Incorrect roll numbers found: {incorrect_roll_df}")

    # Save incorrect roll numbers to a separate Excel file for download
    incorrect_roll_filename = os.path.join(output_folder, 'incorrect_roll_numbers.xlsx')
    incorrect_roll_df.to_excel(incorrect_roll_filename, index=False)

    # Print for debugging
    # print(f"Incorrect roll numbers saved to: {incorrect_roll_filename} with {len(incorrect_roll_df)} records.")
    # Save incorrect roll numbers to a temporary file for download
    incorrect_roll_filename = os.path.join(output_folder, 'incorrect_roll_numbers.xlsx')
    incorrect_roll_df.to_excel(incorrect_roll_filename, index=False)



    # Process question columns
    columns = expected_columns
    counts = {option: 0 for option in ['A', 'B', 'C', 'D', '*', '-']}
    for col in columns:
        for option in counts.keys():
            counts[option] += (df[col] == option).sum()

    # Plotting
    fig_value = px.bar(
        x=list(counts.keys()),
        y=list(counts.values()),
        title=f"Question Pattern: {question_pattern} Questions - Value Distribution",
        labels={'x': 'Options', 'y': 'Count'},
        color=list(counts.keys())
    )
    plot_value_filename = f"value_distribution_{question_pattern}_questions.html"
    pio.write_html(fig_value, file=os.path.join(output_folder, plot_value_filename), auto_open=False)

    counts_exceeding = {
        '*': (df[columns].apply(lambda row: (row == '*').sum(), axis=1) > limit).sum(),
        '-': (df[columns].apply(lambda row: (row == '-').sum(), axis=1) > limit).sum()
    }
    fig_limit = px.bar(
        x=list(counts_exceeding.values()),
        y=list(counts_exceeding.keys()),
        orientation='h',
        title=f"Sheets Exceeding Limit of {limit} for '*' and '-'",
        labels={'x': 'Count', 'y': 'Options'}
    )
    plot_limit_filename = f"horizontal_plot_limit_{limit}.html"
    pio.write_html(fig_limit, file=os.path.join(output_folder, plot_limit_filename), auto_open=False)

    # Confusion scores for pie chart
    confusion_scores = {col: df[col].isna().sum() + (df[col] == '*').sum() + (df[col] == '-').sum() for col in columns}
    top_confusing_questions = sorted(confusion_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    questions, scores = zip(*top_confusing_questions)

    fig_pie = px.pie(
        names=questions,
        values=scores,
        title='Top 10 Most Confusing Questions',
        labels={'names': 'Questions', 'values': 'Scores'}
    )
    pie_chart_filename = f"confusing_questions_pie_chart_{question_pattern}.html"
    pio.write_html(fig_pie, file=os.path.join(output_folder, pie_chart_filename), auto_open=False)

    # Generate demo answer sheet
    demo_answer_sheet = generate_demo_answer_sheet(df, columns)  # Call it once
    demo_answer_sheet_df = pd.DataFrame(demo_answer_sheet, index=[0])
    demo_answer_sheet_html = demo_answer_sheet_df.to_html(index=False)

####################################################3easy_hard#########################
    # Get the easy and hard question counts from the request body
    easy_count = int(request.form.get('easy_count', 3))  # Default is 3
    hard_count = int(request.form.get('hard_count', 3))  # Default is 3

    # Ensure easy_count and hard_count are within the range of 2 to 10
    easy_count = min(max(easy_count, 2), 10)
    hard_count = min(max(hard_count, 2), 10)

    # Calculate difficulty for each question
    difficulty = calculate_difficulty(df)

    # Filter questions to only include those matching the pattern Q1, Q2, etc.
    questions_only = {q: v for q, v in difficulty.items() if re.match(r'Q\d+', q)}

    # Sort questions by difficulty (easy = lower score, hard = higher score)
    sorted_difficulty = sorted(questions_only.items(), key=lambda x: x[1])

    # Get top 'easy_count' easiest questions and top 'hard_count' hardest questions
    easiest_questions = sorted_difficulty[:easy_count]
    hardest_questions = sorted_difficulty[-hard_count:]

    # Create lists for the chart
    question_labels = [q[0] for q in easiest_questions + hardest_questions]
    easy_scores = [q[1] for q in easiest_questions]
    hard_scores = [q[1] for q in hardest_questions]

    # Calculate percentages for the graph
    max_easy_score = max(easy_scores) if easy_scores else 1
    max_hard_score = max(hard_scores) if hard_scores else 1

    easy_scores_percentage = [(score / max_easy_score) * 100 for score in easy_scores]
    hard_scores_percentage = [(score / max_hard_score) * 100 for score in hard_scores]

    # Plotly graph for easy and hard questions (Stacked Bar Chart)
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=question_labels[:easy_count],
        y=easy_scores_percentage,
        name='Easiest Questions',
        marker_color='green'
    ))

    fig.add_trace(go.Bar(
        x=question_labels[easy_count:],
        y=hard_scores_percentage,
        name='Hardest Questions',
        marker_color='red'
    ))

    fig.update_layout(
        title='Easy and Hard Questions Based on Response Distribution',
        xaxis_title='Questions',
        yaxis_title='Difficulty Score (Percentage)',
        barmode='stack',
        legend=dict(x=0, y=1.0),
        margin=dict(l=40, r=0, t=40, b=30)
    )

    # Generate HTML representation of the plot
    easy_hard_graph_html = plot(fig, output_type='div')


    # Metrics to be displayed on the HTML page
    metrics = {
        "Total Number of OMR Sheets": total_sheets,
        "Correct Barcodes": correct_barcodes,
        "Incorrect Barcodes": incorrect_barcodes,
        "Null (Not Predicted) Barcodes": null_barcodes_sum,
        "Correct Predicted Roll Numbers": correct_roll_nos,
        "Incorrect Predicted Roll Numbers": incorrect_roll_nos,
        "Not Predicted Roll Numbers": null_roll_nos
    }

    # Render template and pass the generated plot HTML
    return render_template('charts_page.html',
                        metrics=metrics,
                        easy_hard_graph_html=easy_hard_graph_html,
                        chart_value_filename=plot_value_filename,
                        chart_limit_filename=plot_limit_filename,
                        pie_chart_filename=pie_chart_filename,
                        correct_excel=correct_excel,
                        incorrect_excel=incorrect_excel,
                        null_barcodes_excel='null_barcodes.xlsx',  # Add this line
                        limit=limit,
                        demo_answer_sheet_html=demo_answer_sheet_html,
                        incorrect_roll_filename='incorrect_roll_numbers.xlsx')


@app.route('/download_incorrect_roll_numbers')
def download_incorrect_roll_numbers():
    # Path to the Excel file for incorrect roll numbers
    incorrect_excel_path = os.path.join(app.config['UPLOAD_FOLDER'], "Excel_folder", "incorrect_roll_numbers.xlsx")
    
    # Check if the file exists
    if not os.path.exists(incorrect_excel_path):
        return "File not found!", 404

    # Send the file to the user for download
    return send_file(incorrect_excel_path, as_attachment=True)


@app.route('/download_null_barcodes')
def download_null_barcodes():
    # Path to the Excel file for null (not predicted) barcodes
    null_barcodes_excel_path = os.path.join(app.config['UPLOAD_FOLDER'], "Excel_folder", "null_barcodes.xlsx")
    
    # Check if the file exists
    if not os.path.exists(null_barcodes_excel_path):
        return "File not found!", 404

    # Send the file to the user for download
    return send_file(null_barcodes_excel_path, as_attachment=True)




def format_demo_answer_sheet(df):
    questions = df.columns.tolist()
    answers = df.values.flatten().tolist()
    formatted_lines = []

    # Loop through the questions and answers in chunks of 25
    for i in range(0, len(questions), 25):
        # Create a line for questions and answers
        question_line = "\t".join(questions[i:i + 25])  # Join questions with a tab
        answer_line = "\t".join(answers[i:i + 25])      # Join answers with a tab

        # Combine question and answer lines into one string
        formatted_lines.append(f"<strong>{question_line}</strong><br>{answer_line}")

    # Return the formatted lines joined with HTML line breaks for rendering
    return "<br><br>".join(formatted_lines)  # Two <br> for spacing between each set

@app.route('/update_pie_chart', methods=['POST'])
def update_pie_chart():
    if 'data' not in session:
        return redirect(url_for('upload_file'))
    
    df = pd.DataFrame(session['data'])
    
    # Retrieve the limit from the JSON request
    limit = int(request.json.get('limit', 10))  # Get the updated limit from the JSON body
    # print(f"Received limit: {limit}")  # Debugging statement

    # Your existing logic to calculate confusion scores for the pie chart
    columns = [f'Q{i}' for i in range(1, 31)]  # Adjust based on your question pattern
    confusion_scores = {}
    for col in columns:
        confusion_score = df[col].isna().sum() + (df[col] == '*').sum() + (df[col] == '-').sum()
        confusion_scores[col] = confusion_score

    # Use the dynamic limit to select the top confusing questions
    top_confusing_questions = sorted(confusion_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    # Proceed only if there are confusing questions
    if not top_confusing_questions:
        return jsonify({'error': 'No confusing questions found.'}), 400

    questions, scores = zip(*top_confusing_questions)

    fig_pie = px.pie(
        names=questions,
        values=scores,
        title='Top Confusing Questions',
        labels={'names': 'Questions', 'values': 'Scores'}
    )

    pie_chart_filename = f"confusing_questions_pie_chart_{limit}.html"
    pie_chart_filepath = os.path.join(app.config['UPLOAD_FOLDER'], pie_chart_filename)
    pio.write_html(fig_pie, file=pie_chart_filepath, auto_open=False)

    return jsonify({'pie_chart_filename': pie_chart_filename})  # Return the filename as JSON

@app.route('/update_bar_chart', methods=['POST'])
def update_bar_chart():
    if 'data' not in session:
        return redirect(url_for('upload_file'))
    
    # Load data from session
    df = pd.DataFrame(session['data'])
    
    # Retrieve the limit from the JSON request
    limit = int(request.json.get('limit', 10))

    # Define columns to check
    columns = [f'Q{i}' for i in range(1, 31)]
    
    # Calculate counts exceeding the limit
    counts_exceeding = {
        '*': (df[columns].apply(lambda row: (row == '*').sum(), axis=1) > limit).sum(),
        '-': (df[columns].apply(lambda row: (row == '-').sum(), axis=1) > limit).sum()
    }

    # Create horizontal bar chart for counts exceeding limit
    fig_limit = px.bar(
        x=list(counts_exceeding.values()),
        y=list(counts_exceeding.keys()),
        orientation='h',
        title=f"Sheets Exceeding Limit of {limit} for '*' and '-'",
        labels={'x': 'Count', 'y': 'Options'}
    )

    # Save bar chart as HTML
    bar_chart_filename = f"horizontal_plot_limit_{limit}.html"
    bar_chart_filepath = os.path.join(app.config['UPLOAD_FOLDER'], bar_chart_filename)
    pio.write_html(fig_limit, file=bar_chart_filepath, auto_open=False)

    # Generate separate Excel files for '*' and '-' exceeding the limit
    star_exceeding_df = df[df[columns].apply(lambda row: (row == '*').sum() > limit, axis=1)]
    dash_exceeding_df = df[df[columns].apply(lambda row: (row == '-').sum() > limit, axis=1)]
    
    # Define filenames for each case
    star_excel_filename = f"exceeding_star_limit_{limit}.xlsx"
    dash_excel_filename = f"exceeding_dash_limit_{limit}.xlsx"
    star_excel_filepath = os.path.join(app.config['UPLOAD_FOLDER'], star_excel_filename)
    dash_excel_filepath = os.path.join(app.config['UPLOAD_FOLDER'], dash_excel_filename)

    # Save filtered data to separate Excel files
    star_exceeding_df.to_excel(star_excel_filepath, index=False)
    dash_exceeding_df.to_excel(dash_excel_filepath, index=False)

    # Return filenames in JSON response
    return jsonify({
        'bar_chart_filename': bar_chart_filename,
        'star_excel_filename': star_excel_filename,
        'dash_excel_filename': dash_excel_filename
    })


@app.route('/update_difficulty_charts', methods=['POST'])
def update_difficulty_charts():
    if 'data' not in session:
        return redirect(url_for('upload_file'))

    df = pd.DataFrame(session['data'])
    data = request.get_json()
    
    easy_count = int(data.get('easy_count', 3))
    hard_count = int(data.get('hard_count', 3))

    # Ensure easy_count and hard_count are within the range of 2 to 10
    easy_count = min(max(easy_count, 2), 10)
    hard_count = min(max(hard_count, 2), 10)

    # Calculate difficulty for each question
    difficulty = calculate_difficulty(df)

    # Filter questions to only include those matching the pattern Q1, Q2, etc.
    questions_only = {q: v for q, v in difficulty.items() if re.match(r'Q\d+', q)}

    # Sort questions by difficulty (easy = lower score, hard = higher score)
    sorted_difficulty = sorted(questions_only.items(), key=lambda x: x[1])

    # Get top 'easy_count' easiest questions and top 'hard_count' hardest questions
    easiest_questions = sorted_difficulty[:easy_count]
    hardest_questions = sorted_difficulty[-hard_count:]

    # Create lists for the chart
    question_labels = [q[0] for q in easiest_questions + hardest_questions]
    easy_scores = [q[1] for q in easiest_questions]
    hard_scores = [q[1] for q in hardest_questions]

    # Calculate percentages for the graph
    max_easy_score = max(easy_scores) if easy_scores else 1
    max_hard_score = max(hard_scores) if hard_scores else 1

    easy_scores_percentage = [(score / max_easy_score) * 100 for score in easy_scores]
    hard_scores_percentage = [(score / max_hard_score) * 100 for score in hard_scores]

    # Plotly graph for easy and hard questions (Stacked Bar Chart)
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=question_labels[:easy_count],
        y=easy_scores_percentage,
        name='Easiest Questions',
        marker_color='green'
    ))

    fig.add_trace(go.Bar(
        x=question_labels[easy_count:],
        y=hard_scores_percentage,
        name='Hardest Questions',
        marker_color='red'
    ))

    fig.update_layout(
        title='Easy and Hard Questions Based on Response Distribution',
        xaxis_title='Questions',
        yaxis_title='Difficulty Score (Percentage)',
        barmode='stack',
        legend=dict(x=0, y=1.0),
        margin=dict(l=40, r=0, t=40, b=30)
    )

    easy_hard_graph_html = plot(fig, output_type='div')

    # Return updated chart as JSON
    return jsonify({'easy_hard_graph_html': easy_hard_graph_html})



# FlaskThread to run the Flask server
class FlaskThread(QThread):
    output_signal = Signal(str)

    def run(self):
        try:
            # Start the Flask app
            self.output_signal.emit("Flask server is starting...")
            app.run(debug=False, host='0.0.0.0', port=5000)  # Start Flask server

        except Exception as e:
            self.output_signal.emit(f"Failed to start Flask server: {str(e)}")

class ScriptLauncher(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("OMR")
        self.setGeometry(0, 0, 400, 100)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        name_label = QLabel("OMR SHARP", self)
        name_label.setStyleSheet("color: lightyellow; background-color: purple;")
        name_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        font = name_label.font()
        font.setPointSize(15)
        font.setBold(True)
        name_label.setFont(font)

        layout.addWidget(name_label)
     
        self.input_folder_path_label, input_folder_button,self.input_folder_path= self.add_input_folder(layout,"Input Folder-", "Browse", "Enter Image Folder Path")
        input_folder_button.setFixedSize(200, 20)
        self.input_folder_path.setFixedSize(400, 20)
        input_folder_button.clicked.connect(self.browse_input_folder1)

        self.output_folder_path_label, output_folder_button,self.output_folder_path= self.add_output_folder(layout,"output Folder-", "Browse", "Enter output Folder Path")
        output_folder_button.setFixedSize(200, 20)
        self.output_folder_path.setFixedSize(400, 20)
        output_folder_button.clicked.connect(self.browse_output_folder1)  
        

        self.template_label, template_button, self.template_path1 = self.add_temp_folder(layout,"Reference Image for finding similarity -", "Browse","Enter template path or use a default reference")
        template_button.setFixedSize(200, 20)
        self.template_path1.setFixedSize(400, 20)
        template_button.clicked.connect(self.browse_template_path1)   


        
        dimension_layout = QHBoxLayout()
        self.Image_Size_label=QLabel("Image Size:")
        self.image_Width_label = QLabel("Width", self)
        self.image_Width = QLineEdit(self)
        self.image_Width.setText("1654")

        self.image_Height_label = QLabel("Height", self)
        self.image_Height = QLineEdit(self)
        self.image_Height.setText("2366")

        self.Image_Size_button = QPushButton('Image Size')
        self.Image_Size_button .setStyleSheet("background-color: darkCyan; color: white;")
        self.Image_Size_button.clicked.connect(self.find_image_size)
        
        dimension_layout.addWidget(self.Image_Size_label )
        dimension_layout.addWidget(self.image_Width_label)
        dimension_layout.addWidget(self.image_Width)
        dimension_layout.addWidget(self.image_Height_label)
        dimension_layout.addWidget(self.image_Height)
        dimension_layout.addWidget(self.Image_Size_button)
        layout.addLayout(dimension_layout)
       

        
       
        straight_image_layout = QHBoxLayout()
        

        


        self.pbar = QProgressBar(self)
        self.pbar.setFixedSize(600, 20)
        self.pbar.setRange(0, 10)
        straight_image_layout.addWidget(self.pbar)

        self.Processing = QLineEdit(self)
        straight_image_layout.addWidget(self.Processing)

        self.submitbtn1 = QPushButton(" straight image ", self)
        self.submitbtn1.clicked.connect(self.process_straight_one_image)
        self.submitbtn1.setStyleSheet("background-color: darkCyan; color: white;")
        straight_image_layout.addWidget(self.submitbtn1)

       
        layout.addLayout(straight_image_layout)
                    
        
      

       

        
        self.template_label1 = QLabel("Cordinates points selection -", self)
        layout.addWidget(self.template_label1)
        self.submitbtn = QPushButton("Select Co-Ordinates", self)
        self.submitbtn.clicked.connect(self.threaded_open_html_page)
        self.submitbtn.setStyleSheet("background-color: darkCyan; color: white;")
        layout.addWidget(self.submitbtn)

        self.cordinates_label, cordinate_button, self.cordinate_path = self.add_input_group_coordinates(layout, "Cordinates File -", "Browse", "Upload Cordinates File")
        cordinate_button.setFixedSize(200, 20)
        self.cordinate_path.setFixedSize(400, 20)
        cordinate_button.clicked.connect(self.browse_cordinate_path)
        self.Image_Size_button1 = QPushButton('OMR Pattern ')
        self.Image_Size_button1 .setStyleSheet("background-color: darkCyan; color: white;")
        self.Image_Size_button1.clicked.connect(self.cordinates_point_text_file)
        layout.addWidget(self.Image_Size_button1)
        
        omr = QHBoxLayout()
        self.omr_pattern= QLabel(" OMR Details ", self)
        omr.addWidget(self.omr_pattern) 
        self.columns_platform =QLineEdit() #QComboBox()
        omr.addWidget(self.columns_platform) 
        layout.addLayout(omr) 

        omr_type = QHBoxLayout()
        self.omr_question= QLabel(" OMR Type ", self)
        omr_type.addWidget(self.omr_question) 
        self.columns_platform_question =QLineEdit() # QComboBox()
        omr_type.addWidget(self.columns_platform_question)
        layout.addLayout(omr_type)
        
        ocr_type = QHBoxLayout()
        self.ocr_question= QLabel(" OCR Details ", self)
        ocr_type.addWidget(self.ocr_question) 
        self.columns_platform_ocr =QLineEdit() # QComboBox()
        ocr_type.addWidget(self.columns_platform_ocr)
        layout.addLayout(ocr_type)

        self.sift_label= QLabel("Features Selection", self)
        layout.addWidget(self.sift_label) 
        self.list_widget_sift = QListWidget()
        self.list_widget_sift.addItems(["0","2000", "2500", "3000", ])
        self.list_widget_sift.setSelectionMode(QListWidget.MultiSelection)
        self.list_widget_sift.itemSelectionChanged.connect(self.update_sift_options)
        layout.addWidget(self.list_widget_sift)
        self.columns_platform_sift = QComboBox()
        layout.addWidget(self.columns_platform_sift)

        
        # self.cb = QComboBox()
        # self.cb.addItems(["0","2000", "3000","5000"])
        # #self.cb.currentIndexChanged.connect(self.selectionchange)
        # layout.addWidget(self.cb)

        image_layout = QHBoxLayout()
        self.similarity_per_label = QLabel("Similarity Percentage -", self)
        self.similarity_per = QLineEdit(self)
        self.similarity_per.setPlaceholderText("Enter similarity percentage")
        self.similarity_per.setText("0")  # Set default value
        

        self.table_label = QLabel("OMR(Table name) -", self)
        layout.addWidget(self.table_label)
        self.table_value = QLineEdit(self)
        self.table_value.setPlaceholderText("Enter table name")
        
            
        self.admit_card_roll = QLabel("Admit Card(Table Name)  -", self)

        self.admit_card_roll_value = QLineEdit(self)
        self.admit_card_roll_value.setPlaceholderText("Enter Database Table")

       

        image_layout.addWidget( self.similarity_per_label)
        image_layout.addWidget(self.similarity_per)
        image_layout.addWidget( self.table_label )
        image_layout.addWidget(self.table_value)
        image_layout.addWidget( self.admit_card_roll  )
        image_layout.addWidget(self.admit_card_roll_value)
        layout.addLayout(image_layout)
        
        # self.table_label2 = QLabel("Incorrect Table Name -", self)
        # layout.addWidget(self.table_label2)
        # self.table_value2 = QLineEdit(self)
        # self.table_value2.setPlaceholderText("Enter table name")


        # image_layout.addWidget( self.table_label2 )
        # image_layout.addWidget(self.table_value2)

        # self.Admit_card_label= QLabel("Admit Card Fields", self)
        # layout.addWidget(self.Admit_card_label) 
        # self.list_widget_admit = QListWidget()
        # self.list_widget_admit.addItems(["Booklet_No_0_9","Booklet_No_1_0", "College_Code","Exam_Date",
        #                                 "Registration","Roll_No_0_9","Roll_No_1_0" ,"Subject_Code" ])
        # self.list_widget_admit.setSelectionMode(QListWidget.MultiSelection)
        # self.list_widget_admit.itemSelectionChanged.connect(self.update_admit_column_options)
        # layout.addWidget(self.list_widget_admit)
        # self.columns_platform_admit = QComboBox()
        # layout.addWidget(self.columns_platform_admit)

         
        # admit_layout = QHBoxLayout()   
        
        # self.admit_card_roll_column = QLabel("Admit Card Coulmn Name  -", self)

        # self.admit_card_roll_column_value = QLineEdit(self)
        # self.admit_card_roll_column_value.setPlaceholderText("Enter Database Table")

        # admit_layout.addWidget( self.admit_card_roll_column  )
        # admit_layout.addWidget(self.admit_card_roll_column_value)
        # layout.addLayout(admit_layout)
        
        self.db_label, db_button, self.db_path = self.add_input_group(layout, "SQLite Path -", "Browse", "Enter database path")
        db_button.setFixedSize(200, 20)
        self.db_path.setFixedSize(400, 20)
        db_button.clicked.connect(self.browse_db_path1)

        self.template_label2, template_button2, self.template_path2 = self.add_input_group_template(layout, "Reference Image  -", "Browse", "Enter template path or use a default reference")
        template_button2.setFixedSize(200, 20)
        self.template_path2.setFixedSize(400, 20)
        template_button2 .clicked.connect(self.browse_template_path2)
        
        


        self.progress_label = QLabel("Batch Processing Result", self)
        layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)

        self.BatchProcessing_label_path = QLabel(self)
        layout.addWidget(self.BatchProcessing_label_path)

        time_layout = QHBoxLayout()
        self.All_image_time_label = QLabel("  Remaining  Time -", self)
        self.All_image_time_value = QLineEdit(self)
        self.All_image_time_value.setText("0:0")  # Set default value
        #self.one_image_time_value.setFixedSize(200, 20)
        self.All_image_time_label .setStyleSheet("color:blue")
        font = self.All_image_time_label .font()
        font.setPointSize(10)
        font.setBold(True)
        self.All_image_time_label.setFont(font)

        self.All_image_time_value .setStyleSheet("color:red")
        font = self.All_image_time_value.font()
        font.setPointSize(10)
        font.setBold(True)
        self.All_image_time_value.setFont(font)

        self.end_time_label = QLabel(" Total Time Taken -", self)
        self.end_time_value = QLineEdit(self)
        self.end_time_value.setText("0:0") 
        #self.end_time_value.setFixedSize(200, 20)
        self.end_time_label.setStyleSheet("color:blue")
        font = self.end_time_label.font()
        font.setPointSize(10)
        font.setBold(True)
        self.end_time_label.setFont(font)
        self.end_time_value.setStyleSheet("color:red")
        font = self.end_time_value.font()
        font.setPointSize(10)
        font.setBold(True)
        self.end_time_value.setFont(font)
        time_layout.addWidget(self.All_image_time_label)
        time_layout.addWidget(self.All_image_time_value)
        time_layout.addWidget(self.end_time_label)
        time_layout.addWidget(self.end_time_value)
        layout.addLayout(time_layout)

        # self.url_button = QPushButton('Get Flask URL')
        # self.url_button.clicked.connect(self.show_url)
        # layout.addWidget(self.url_button)

        # self.run_flask_button = QPushButton('Run Flask Server')
        # self.run_flask_button.clicked.connect(self.run_flask)
        # layout.addWidget(self.run_flask_button)
        
        # self.log_output = QTextEdit()
        # self.log_output.setReadOnly(True)
        # layout.addWidget(self.log_output)

        # self.flask_thread = FlaskThread()
        # self.flask_thread.output_signal.connect(self.append_log)


        self.submitbtn = QPushButton("Start", self)
        self.submitbtn.setFixedSize(200, 30)
        layout.addStretch()
        self.submitbtn.clicked.connect(self.gui_variable)  
        self.submitbtn.setStyleSheet("background-color: green; color: white;")

        closebtn = QPushButton("Close", self)
        closebtn.setFixedSize(200, 30)
        closebtn.clicked.connect(self.close_window)
        closebtn.setStyleSheet("background-color: red; color: white;")

        
        # Analyzerbtn = QPushButton("Analyzer", self)
        # Analyzerbtn.setFixedSize(200, 30)
        # Analyzerbtn.clicked.connect(self.call_flash_from_omr)
        # Analyzerbtn.setStyleSheet("background-color: red; color: white;")




        # self.run_flask_button = QPushButton('Run Flask Server')
        # self.run_flask_button.setFixedSize(200, 30)
        # layout.addStretch()
        # self.run_flask_button.clicked.connect(self.run_flask)
        # self.run_flask_button.setStyleSheet("background-color:orange; color: white;")
        # #layout.addWidget(self.run_flask_button)

        # # Create a text area to display logs
        # self.log_output = QTextEdit()
        # self.log_output.setReadOnly(True)
        # layout.addWidget(self.log_output)
        # self.setLayout(layout)
        # self.resize(400, 300)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.submitbtn)
        button_layout.addWidget(closebtn)
        #button_layout.addWidget(Analyzerbtn)
       
        #button_layout.addWidget(self.run_flask_button)

        layout.addLayout(button_layout)


   
        self.url_button = QPushButton('Analyze OMR Result')
        self.url_button.setFixedSize(300, 30)
        self.url_button.clicked.connect(self.show_url)
        self.url_button.setStyleSheet("background-color:lightBlue; color: white;")

        # Create a horizontal layout to center the button
        button_layout = QHBoxLayout()
        button_layout.addStretch()  # Add stretchable space before the button
        button_layout.addWidget(self.url_button)  # Add the button in the center
        button_layout.addStretch()  # Add stretchable space after the button

        # Add the button layout to the main layout
        layout.addLayout(button_layout)

    
    # def selectionchange(self):
    #     global nfeatures
    #     for count in range(self.cb.count()):
    #         nfeatures= self.cb.itemText(count)
    #         nfeatures_list.append(nfeatures)
        
    #     print("features",nfeatures)
    
    def show_url(self):
        # Find local IP address
        self.flask_thread = FlaskThread()
        
        self.flask_thread.output_signal.connect(self.append_log)
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        self.url = f"http://{ip_address}:5000"
        
        # Show URL in button
        #self.url_button.setText(f"Open {self.url}")
        self.url_button.setText(f" Click to open OMR Analyzer")
        self.url_button.clicked.disconnect()  # Disconnect previous signal
        self.url_button.clicked.connect(lambda: webbrowser.open(self.url))  # Connect new signal
        self.flask_thread.start()
        self.append_log("Flask server is starting...")

        # Enable the Flask server button
        #self.run_flask_button.setEnabled(True)

    # def run_flask(self):
    #     # Start the Flask server in a separate thread
    #     self.flask_thread.start()
    #     self.append_log("Flask server is starting...")

    def append_log(self, message):
        self.log_output.append(message)
    def find_image_size(self):
        folder_path= self.input_folder_path.text()
        filename_list=os.listdir(folder_path)
        filename=filename_list[0]
        image_path = os.path.join(folder_path, filename)
        image=cv2.imread(image_path)
        height,width,_=image.shape
        self.image_Width.setText(f'{width}')
        self.image_Height.setText(f'{height}')
        
    def gui_variable(self):
        # Retrieve values from GUI inputs
        self.table_name = self.table_value.text()
        #self.table_name2 = self.table_value2.text()
        self.template_path=self.template_path2.text()
        self.db_path_value= self.db_path.text()
        self.folder_path = self.input_folder_path.text()
        self.output_path_final = self.output_folder_path.text()
        self.cordinates_point_path = self.cordinate_path.text()
        #self.admit_roll=self.admit_card_roll_value.text()
        self.admit_card_database_table = self.admit_card_roll_value.text()
        #self.admit_card_database_column_name=self.admit_card_roll_column_value.text()
        
       
        try:
            # Check if all required fields are filled
            if all([self.table_name,self.template_path,self.db_path_value,self.folder_path,self.output_path_final,self.cordinates_point_path]):
                self.thread_calling()
            else:

                QMessageBox.information(self, "Input Error", "Please fill in all required fields.")
        except Exception as e:
            QMessageBox.information(self, "Error", f"Database Entry Error: {e}")
            self.refresh()

    # def update_columns_options(self):
    #     selected_items = [item.text() for item in self.list_widget.selectedItems()]
    #     self.columns_platform.clear()
    #     if selected_items:
    #         combined_text = ', '.join(selected_items)
    #         # Limit the length of the combined text
    #         max_length = 200
    #         ellipsis = '...'
    #         if len(combined_text) > max_length:
    #             combined_text = combined_text[:max_length - len(ellipsis)] + ellipsis
    #         #self.columns_platform.addItem(combined_text)
    #         self.columns_platform.setText(combined_text)
    def update_question_options(self):
        selected_items = [item.text() for item in self.list_widget_question.selectedItems()]
        self.columns_platform_question.clear()
        if selected_items:
            combined_text = ' , '.join(selected_items)
            # Limit the length of the combined text
            max_length = 100
            ellipsis = '...'
            if len(combined_text) > max_length:
                combined_text = combined_text[:max_length - len(ellipsis)] + ellipsis
            #self.columns_platform_question.addItem(combined_text) 
            self.columns_platform_question.setText(combined_text)     

    def update_sift_options(self):
        selected_items = [item.text() for item in self.list_widget_sift.selectedItems()]
        self.columns_platform_sift.clear()
        if selected_items:
            combined_text = ' , '.join(selected_items)
            # Limit the length of the combined text
            max_length = 100
            ellipsis = '...'
            if len(combined_text) > max_length:
                combined_text = combined_text[:max_length - len(ellipsis)] + ellipsis
            self.columns_platform_sift.addItem(combined_text)   
    
    def update_admit_column_options(self):
        selected_items = [item.text() for item in self.list_widget_admit.selectedItems()]
        self.columns_platform_admit.clear()
        if selected_items:
            combined_text = ' , '.join(selected_items)
            # Limit the length of the combined text
            max_length = 100
            ellipsis = '...'
            if len(combined_text) > max_length:
                combined_text = combined_text[:max_length - len(ellipsis)] + ellipsis
            self.columns_platform_admit.addItem(combined_text)   
            
    def add_input_folder(self, layout, label_text, button_text, placeholder_text):
        input_layout = QHBoxLayout()

        label = QLabel(label_text, self)
        input_layout.addWidget(label)

        button = QPushButton(button_text, self)
        input_layout.addWidget(button)

        line_edit = QLineEdit(self)
        line_edit.setPlaceholderText(placeholder_text)
        input_layout.addWidget(line_edit)

        layout.addLayout(input_layout)

        return label, button, line_edit
    
    def add_output_folder(self, layout, label_text, button_text, placeholder_text):
        input_layout = QHBoxLayout()

        label = QLabel(label_text, self)
        input_layout.addWidget(label)

        button = QPushButton(button_text, self)
        input_layout.addWidget(button)

        line_edit = QLineEdit(self)
        line_edit.setPlaceholderText(placeholder_text)
        input_layout.addWidget(line_edit)

        layout.addLayout(input_layout)

        return label, button, line_edit

    # def add_temp_folder(self, layout, label_text, button_text, placeholder_text):
    #     input_layout = QHBoxLayout()

    #     label = QLabel(label_text, self)
    #     input_layout.addWidget(label)

    #     button = QPushButton(button_text, self)
    #     input_layout.addWidget(button)

    #     line_edit = QLineEdit(self)
    #     line_edit.setPlaceholderText(placeholder_text)
    #     input_layout.addWidget(line_edit)

    #     layout.addLayout(input_layout)

    #     return label, button, line_edit 

    def add_temp_folder(self, layout, label_text, button_text, placeholder_text):
        input_layout = QHBoxLayout()

        label = QLabel(label_text, self)
        input_layout.addWidget(label)

        button = QPushButton(button_text, self)
        input_layout.addWidget(button)

        line_edit = QLineEdit(self)
        line_edit.setPlaceholderText(placeholder_text)
        input_layout.addWidget(line_edit)

        self.b1 = QCheckBox("Resize")
        self.b1.setChecked(False)  # Set the checkbox to default to checked
        self.b1.stateChanged.connect(lambda: self.btnstate(self.b1))
        input_layout.addWidget(self.b1)
     

        layout.addLayout(input_layout)

        return label, button, line_edit 


    def add_input_group(self, layout, label_text, button_text, placeholder_text):
        input_layout = QHBoxLayout()

        label = QLabel(label_text, self)
        input_layout.addWidget(label)

        button = QPushButton(button_text, self)
        input_layout.addWidget(button)

        line_edit = QLineEdit(self)
        line_edit.setPlaceholderText(placeholder_text)
        input_layout.addWidget(line_edit)

        layout.addLayout(input_layout)

        return label, button, line_edit
    
    def add_input_group_template(self, layout, label_text, button_text, placeholder_text):
        input_layout = QHBoxLayout()

        label = QLabel(label_text, self)
        input_layout.addWidget(label)

        button = QPushButton(button_text, self)
        input_layout.addWidget(button)

        line_edit = QLineEdit(self)
        line_edit.setPlaceholderText(placeholder_text)
        input_layout.addWidget(line_edit)

        layout.addLayout(input_layout)

        return label, button, line_edit
    
   
    def add_input_group_coordinates(self, layout, label_text, button_text, placeholder_text):
        input_layout = QHBoxLayout()

        label = QLabel(label_text, self)
        input_layout.addWidget(label)

        button = QPushButton(button_text, self)
        input_layout.addWidget(button)

        line_edit = QLineEdit(self)
        line_edit.setPlaceholderText(placeholder_text)
        input_layout.addWidget(line_edit)
        layout.addLayout(input_layout)

        return label, button, line_edit
    
    def input_admit_card(self, layout, label_text, button_text, placeholder_text):
        input_layout = QHBoxLayout()

        label = QLabel(label_text, self)
        input_layout.addWidget(label)

        button = QPushButton(button_text, self)
        input_layout.addWidget(button)

        line_edit = QLineEdit(self)
        line_edit.setPlaceholderText(placeholder_text)
        input_layout.addWidget(line_edit)

        layout.addLayout(input_layout)

        return label, button, line_edit
  
    def thread_calling(self):

        
        self.submitbtn.setDisabled(True)

        self.worker_thread = WorkerThread(self)
        self.worker_thread.start()
        self.worker_thread.omr_remaining_time.connect(self.omr_remaining_time)
        self.worker_thread.end_timer.connect(self.omr_end_timer)
        self.worker_thread.finished.connect(self.result_done)
        self.worker_thread.update_progress.connect(self.batch_progress)
        self.worker_thread.db_error.connect(self.db_entry_error)

    def omr_remaining_time(self,time_left):
        self.All_image_time_value.setText(f'{time_left}')

    def omr_end_timer(self,total_time_hours):
        self.end_time_value.setText(f'{total_time_hours}')    
        
    def db_entry_error(self,db_connection_error):    
            QMessageBox.information(self, "Error", f"Database Connection Error: {db_connection_error}" )
            self.refresh_omr()
            #self.close()
    
         
    def result_done(self,error_result):
        if error_result==0:
            QMessageBox.information(self, "Result", "OMR Result Prepared")
        self.refresh_omr()
        self.submitbtn.setDisabled(False)
           
    def refresh(self):
        global error_result
        error_result=0
        # template_path=''
        # folder_path=''
        # output_path_final=''

    def refresh_omr(self):
        global total_image_length
        total_image_length=0

    def threaded_open_html_page(self):
        def open_html_file(file_path):
            abs_path = os.path.abspath(file_path)
            url = f'file://{abs_path}'
            webbrowser.open(url)

        if __name__ == '__main__':
            open_html_file(html_path)
            

    def batch_progress(self, val):
        
        self.progress_bar.setMaximum(total_image_length)
        self.progress_bar.setValue(val)
        self.batch_processing_function(total_image_length,val)
        
    def batch_processing_function(self,total_image_length,val):
        
        text = f"Processing Batch {val}/{total_image_length}"
        self.BatchProcessing_label_path.setText(text)
  
 
    
    
    def  browse_html_file(self):
        filename1, _ = QFileDialog.getOpenFileName(self, 'Select HTML INDEX File', filter='Image Files (*.html)')
        self.html_file_path.setText(filename1)

    def browse_input_folder1(self):
        filename = QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.input_folder_path.setText(filename)

    def browse_template_path1(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Select Image File', filter='Image Files (*.jpg)')
        self.template_path1.setText(filename)
       
    def browse_output_folder1(self):
        filename = QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.output_folder_path.setText(filename)
    def browse_template_path2(self):
        filename1, _ = QFileDialog.getOpenFileName(self, 'Select Image File', filter='Image Files (*.jpg)')
        self.template_path2.setText(filename1)
    def browse_cordinate_path(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Select Cordinates file', filter='cordinates file (*.txt)')
        self.cordinate_path.setText(filename)
    def browse_admit_card(self):
        # Open a file dialog to select the Excel file
        file_dialog = QFileDialog(self)
        filename, _ = file_dialog.getOpenFileName(self, 'Select Excel file', '', 'Excel Files (*.xlsx *.xls)')
        if filename:
            self.admit_card_path.setText(filename)

       
    def browse_db_path1(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Select database File', filter='Image Files (*.db)')
        self.db_path.setText(filename)
    def close_window(self):
        global stop
        stop = 1
        self.close()
    def process_straight_one_image(self):
        
        
        width=self.image_Width.text()
        height=self.image_Height.text()
      
        Image_width=int(width)
        Image_height=int(height)
        

        folder_path= self.input_folder_path.text()
      
        template_path11=self.template_path1.text()
        template_path_SI=template_path11
        output_path = os.path.join(folder_path, "output_straight_image_folder_for_Template")
        os.makedirs(output_path, exist_ok=True)
        template_path11=''
        def straighten_image_10(input_path, template_path_SI):
                print("straighten_image_10 function called")
                image = cv2.imread(input_path)
         
                print("print image path inside straighten_image_10",input_path)
                template_image=cv2.imread(template_path_SI)
                template_image= cv2.resize(template_image,(Image_width,Image_height))
                
                if image is None:
                    print(f"Failed to load image: {input_path}")
                    return None
                    
                                    # Convert to grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Create SIFT detector with a reduced number of features
                sift = cv2.SIFT_create()

                # Detect keypoints and descriptors
                keypoints_template, descriptors_template = sift.detectAndCompute(template_image, None)
                keypoints_image, descriptors_image = sift.detectAndCompute(gray_image, None)

                # Use FLANN-based matcher for faster matching
                index_params = dict(algorithm=1, trees=5)
                search_params = dict(checks=50)  # Fewer checks to speed up matching
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(descriptors_template, descriptors_image, k=2)

                # Apply ratio test to filter good matches
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:  # 0.7 is a common ratio for filtering
                        good_matches.append(m)

                # Check if enough matches are found
                if len(good_matches) < 4:
                    print("Error: Not enough matches found.")
                    return None

                # Extract matched points
                points_template = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                points_image = np.float32([keypoints_image[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Calculate the homography matrix with a relaxed threshold for RANSAC
                transformation_matrix, _ = cv2.findHomography(points_image, points_template, cv2.RANSAC, 3.0)  # Increased threshold

                # Apply the perspective warp
                height, width = image.shape[:2]
                straightened_image = cv2.warpPerspective(image, transformation_matrix, (width, height))

                # Create a mask for valid pixels
                mask = np.where((straightened_image[:, :, 0] != 0) | (straightened_image[:, :, 1] != 0) | (straightened_image[:, :, 2] != 0), 255, 0).astype(np.uint8)
                straightened_image = cv2.bitwise_and(straightened_image, straightened_image, mask=mask)

                # Resize to desired dimensions
                straightened_image = cv2.resize(straightened_image, (Image_width,Image_height))
                 
                return straightened_image


        def Similarity_average_check(straightened_image,template_path_SI ) :
                            template_image=cv2.imread(template_path_SI)
                            template_image= cv2.resize(template_image,(Image_width,Image_height))
                            if straightened_image is  None:
                                print("no image")
                            else:
                                similarity1 = cv2.matchTemplate(template_image, straightened_image, cv2.TM_CCOEFF_NORMED)   
                                similarity_percentage1 = similarity1.max() * 100
                                
                            return similarity_percentage1
        
      

        def similarity_check2(output_path, template_path_SI, image_count):
            global new_output_path
            similarity_filenames = []
            similarity_scores = []

            for filename in os.listdir(output_path):
                if filename.lower().endswith(('.jpg', '.png')):
                    image_path = os.path.join(output_path, filename)
                    image = cv2.imread(image_path)

                    similarity_score = Similarity_average_check(image, template_path_SI)

                    similarity_filenames.append(filename)
                    similarity_scores.append(similarity_score)

            if similarity_scores:
                max_similarity_index = similarity_scores.index(max(similarity_scores))
                max_similarity_filename = similarity_filenames[max_similarity_index]

              
                new_output_path = os.path.join(output_path, "out1")
                os.makedirs(new_output_path, exist_ok=True)

                for filename in os.listdir(output_path):
                    if filename.lower().endswith(('.jpg', '.png')) and max_similarity_filename == filename:
                        image_path = os.path.join(output_path, filename)
                        new_image_path = os.path.join(new_output_path, filename)
                        image = cv2.imread(image_path)
                        cv2.imwrite(new_image_path, image)
            else:
                print("No valid similarity scores.")

            return new_output_path
        
   
        def similarity_check1(straight_image_list, output_path,template_path_SI, final_similarity_average):
                    output_filename=''
                    check = 1
                    i = 0
                  
                    for straightened_image in straight_image_list:
                        if check <= 10:
                           
                            similarity_percentage = Similarity_average_check(straightened_image,template_path_SI)
                            if similarity_percentage > final_similarity_average:
                                output_filename = os.path.join(output_path, f"{i+1}.jpg")
                                cv2.imwrite(output_filename, straightened_image)
                                i += 1
                                check += 1
                         
                    image_files = [
                        file for file in os.listdir(output_path) 
                        if os.path.isfile(os.path.join(output_path, file)) and file.lower().endswith('.jpg')
                    ]
                    image_count=len(image_files)
         
                    if image_count> 1 :
                        image_path=''
                        
                        new_image_path=similarity_check2(output_path,template_path_SI, image_count)
                        for image_file in os.listdir(new_image_path):
                            if image_file.endswith('.jpg'):
                                image_path = os.path.join(new_image_path, image_file)
                       
                        QMessageBox.information(self, "Result", f"Straight Image Saved @ : {new_image_path}")
                        self.template_path2 .setText(f'{image_path }')
                        return 1
                    else:
                     
                        image_path=''
                        filename_straight=[]
                        for image_file in os.listdir(output_path):
                            filename_straight.append(image_file)
                        filename= filename_straight[0]   
                        image_path = os.path.join(output_path, filename)

                        QMessageBox.information(self, "Result", f"Straight Image Saved @ : {output_filename}")
             
                        self.template_path2 .setText(f'{image_path}')
                        return 0
                
        def similarity_check(folder_path, template_path_SI):
            check = 1
            total = 0
            straight_image_list = []
            check_resize = 1
            total_image_checking = 10

            # Check if resizing is enabled
            if Resize_CheckBox == 'True':
                #print("Resize is True")
                
                # Create the resized image folder
                resized_folder = os.path.join(folder_path, "Resized_Folder")
                os.makedirs(resized_folder, exist_ok=True)
                
                # Resize first 10 images
                for filename in os.listdir(folder_path):
                    if filename.endswith('.jpg') or filename.endswith('.png'):
                        image_path = os.path.join(folder_path, filename)
                        img = cv2.imread(image_path)  # Read the image
                        
                        if img is None:
                            print(f"Error: Could not load image {image_path}")
                            continue  # Skip if image is not loaded

                        # Only resize the first 10 images
                        if check_resize == 1:
                            first_image_size = (img.shape[1], img.shape[0])  # Use the size of the first image
                        
                        resize_path = os.path.join(resized_folder, os.path.basename(image_path))
                        resized_img = cv2.resize(img, first_image_size)
                        cv2.imwrite(resize_path, resized_img)
                        
                        print(f"Image saved at {resize_path}")
                        
                        check_resize += 1
                        if check_resize > 10:
                            break  # Stop after resizing 10 images

                # After resizing, update folder path to point to resized folder
                folder_path = resized_folder
            
            
            for filename in os.listdir(folder_path):
             
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    image_path = os.path.join(folder_path, filename)
                    
                    if check <= 10:
                        text_to_display = f"Processing :: {check}/{total_image_checking}"
                        self.Processing.setText(text_to_display)
                        self.pbar.setValue(check)
                        
                        # Straighten the image
                        straightened_image = straighten_image_10(image_path, template_path_SI)
                        
                        # Ensure straightened_image is valid before proceeding
                        if straightened_image is None:
                            print(f"Error: Could not process image {image_path}")
                            continue
                        
                        
                        straight_image_list.append(straightened_image)
                       
                        
                        # Calculate similarity
                        sum_similarity = Similarity_average_check(straightened_image, template_path_SI)
                        total += sum_similarity
                        check += 1

            average = total / 10
            final_similarity_average = average

            return final_similarity_average, straight_image_list


        final_similarity_average, straight_image_list = similarity_check(folder_path, template_path_SI)
       
        QMessageBox.information(self, "Result", f"Average similarity of first 10 images is: {final_similarity_average}")
        simialartity_check1_return=similarity_check1(straight_image_list,output_path,template_path_SI, final_similarity_average)
        final_similarity_average=int(final_similarity_average)-25
        self.similarity_per .setText(f'{final_similarity_average}')
        del folder_path
      
                                
                                
         
    def cordinates_point_text_file(self):
        global  barcode_points11_user,roll_points1_user,roll_points2_user,exam_date_points_user,\
            registration_points_user,college_code_points_user,booklet_no_points1_user,\
            question_paper_Series_Code_points_user,course_points_user,sitting_points_user,stream_horizontal_points_user,stream_vertical_points_user,\
        question_points_40_user,question_points_50_user,question_points_70_user,question_points_100_user,question_points_150_user,question_points_200_user,column_list,\
        subject_code_points_user,category1_points_user,category2_points_user,stream_subcategory_points_user,booklet_code_points_user,\
        gender_points_user,domicle_points_user,disabled_points_user,subject_group_points_user,booklet_no_points_user,\
        series_points_user,faculty_points_user,series_pattern2_user,roll_digit_points,\
        booklet_digit_points_user,college_code_digit_points_user,exam_digit_points_user,reg_digit_points_user,subject_digit_points_user

        global roll_digit_cordinates,booklet_digit_cordinates,college_code_digit_cordinates,exam_digit_cordinates,reg_digit_cordinates,subject_digit_cordinates
        #global roll_type,booklet_type
        roll_type=''
        booklet_type='' 
        
        cordinates_point_path=self.cordinate_path.text()
        column_field=[]
        #db_column_rechecked=[]
        ques_field=[]
        ocr_field=[]
        # roll_type=''
        # booklet_type=''
        def update_global_variables(data):
            for key, value in data.items():
                if key in globals() and isinstance(globals()[key], list):
                    globals()[key].extend(value)
        def clean_and_format_data( data):

                cleaned_data = {}

                for variable_name, variable_data in data.items():
                    cleaned_list = []

                    for entry in variable_data:
                        
                        if all(key in entry for key in ['left', 'top', 'width', 'height']):
                            point_tuple = (
                                entry['left'],
                                entry['top'],
                                entry['width'],
                                entry['height']
                            )
                            cleaned_list.append(point_tuple)

                    cleaned_data[variable_name] = cleaned_list

                return cleaned_data


        with open(cordinates_point_path, "r") as file:
            data_received = json.load(file)

        for key, value in data_received.items():
            if key in globals() and isinstance(globals()[key], list):
                globals()[key].extend(value)
           

        cleaned_data = clean_and_format_data(data_received)
        

            
        for key, value in cleaned_data.items():
            globals()[key] = value
           
            

            if key == 'Barcode':
                barcode_points11_user=[]
                barcode_points11_user.extend(value)
                
                column_field.append("Barcode")
            

            if key == 'Roll_No_0_9':
                roll_type='Roll_No_0_9'
                roll_points1_user=[]
                roll_points1_user.extend(value)
                column_field.append("Roll_No_0_9")
                ocr_field.append('Roll_No_0_9')
                print("Roll_No_0_9", roll_points1_user)

            if key == 'Roll_No_1_0':
                roll_type='Roll_No_1_0'       
                roll_points2_user=[]
                roll_points2_user.extend(value)
                column_field.append("Roll_No_1_0")
                ocr_field.append('Roll_No_1_0')
                

            if key == 'Registration':

                registration_points_user=[]
                registration_points_user.extend(value)
                column_field.append("Registration")
               

            if key == 'College_Code':
                college_code_points_user=[]
                college_code_points_user.extend(value)
                column_field.append("College_Code")

            if key == 'Exam_Date':
                exam_date_points_user=[]
                exam_date_points_user.extend(value)
                column_field.append("Exam_Date")
                    
            if key == 'Sitting':
                sitting_points_user=[]
                sitting_points_user.extend(value)
                column_field.append("Sitting")

            if key == 'Course':
                course_points_user=[]
                course_points_user.extend(value) 
                column_field.append("Course")
 

            if key == 'Question_paper_Series_Code':
                question_paper_Series_Code_points_user=[]
                question_paper_Series_Code_points_user.extend(value) 
                column_field.append("Question_paper_Series_Code")
            
            if key == 'Stream_Horizontal':
                stream_horizontal_points_user=[]
                stream_horizontal_points_user.extend(value)
                column_field.append("Stream_Horizontal")
                
            if key == 'Stream_Vertical':
                stream_vertical_points_user=[]
                stream_vertical_points_user.extend(value)
                column_field.append( "Stream_Vertical")
   

            if key == 'Subject_Code':
                subject_code_points_user=[]
                subject_code_points_user.extend(value)
                column_field.append("Subject_Code")
                
            
            if key =='Category_Pattern_Gen_BC_SC_ST':
                category1_points_user=[]
                category1_points_user.extend(value)
                column_field.append("Category_Pattern_Gen_BC_SC_ST")
                
            
            if key=='Category_Pattern_OBC_UR_SC_ST':
                category2_points_user=[]
                category2_points_user.extend(value)
                column_field.append("Category_Pattern_OBC_UR_SC_ST")

   
                
            if key=='Stream_Others_Vocational_General_Honours':
                stream_subcategory_points_user=[]
                stream_subcategory_points_user.extend(value)
                column_field.append("Stream_Others_Vocational_General_Honours")
                
            if key == 'Booklet_Code_Series':
                booklet_code_points_user=[]
                booklet_code_points_user.extend(value)
                column_field.append("Booklet_Code_Series")
                

            if key == 'Gender':
                gender_points_user=[]
                gender_points_user.extend(value)
                column_field.append("Gender")
                   

            if key == 'Domicile':
                domicle_points_user=[]
                domicle_points_user.extend(value)
                column_field.append("Domicile")
            

            if key == 'Physically_Disabled':
                disabled_points_user=[]
                disabled_points_user.extend(value)
             
                column_field.append("Physically_Disabled")
                
            
            if key == 'Subject_PCM_PCB':
                subject_group_points_user=[]
                subject_group_points_user.extend(value)
                column_field.append("Subject_PCM_PCB")
                             
            
            if key == 'Booklet_No_0_9':
                booklet_no_points1_user=[]
                booklet_type='Booklet_No_0_9'
                booklet_no_points1_user.extend(value)
                column_field.append("Booklet_No_0_9")
                ocr_field.append('Booklet_No_0_9')

            if key == 'Booklet_No_1_0':
                booklet_type='Booklet_No_1_0'
                booklet_no_points_user=[]
                booklet_no_points_user.extend(value)
                column_field.append("Booklet_No_1_0")
                ocr_field.append('Booklet_No_1_0')

            if key == 'Series':
                series_points_user=[]
                series_points_user.extend(value)
                column_field.append("Series")

            if key == 'Series2':
                series_pattern2_user=[]
                series_pattern2_user.extend(value)
                column_field.append("Series2")
            
            if key == 'Faculty':
                faculty_points_user=[]
                faculty_points_user.extend(value)
                column_field.append("Faculty")
    
            if key == 'question_points_40':
                question_points_40_user=[]
                question_points_40_user.extend(value) 
                ques_field.append("40")               
                        
            if key == 'question_points_50':
                question_points_50_user=[]
                question_points_50_user.extend(value)  
                ques_field.append("50")                 
            if key == 'question_points_70':
                question_points_70_user=[]
                question_points_70_user.extend(value) 
                ques_field.append("70")                 
            if key == 'question_points_100':
                question_points_100_user=[]
                question_points_100_user.extend(value)

                ques_field.append("100")   
               
            if key == 'question_points_150':
                question_points_150_user=[]
                question_points_150_user.extend(value)
                ques_field.append("150")   
            if key == 'question_points_200':
                question_points_200_user=[]
                question_points_200_user.extend(value) 
                ques_field.append("200") 

            if key == 'Roll_No_OCR':
                roll_digit_points=[]
                #db_column_rechecked.append(roll_type)
                roll_digit_points.extend(value)
                roll_digit_cordinates=roll_digit_points
                print(roll_digit_points)
                #ocr_field.append(roll_type)
        
            if key == 'Booklet_No_OCR':
                booklet_digit_points_user=[]
                #db_column_rechecked.append(booklet_type)
                booklet_digit_points_user.extend(value)
                booklet_digit_cordinates=booklet_digit_points_user
                print("booklet_digit_points_user",booklet_digit_points_user)
                #ocr_field.append(booklet_type)

            if key == 'College_Code_OCR':
                college_code_digit_points_user=[]
                #db_column_rechecked.append('College_Code')
                college_code_digit_points_user.extend(value)
                college_code_digit_cordinates=college_code_digit_points_user
                print("college_code_digit_points_user",college_code_digit_points_user)
                ocr_field.append('College_Code')
             
            
            if key == 'Exam_Date_OCR':
                exam_digit_points_user=[]
                #db_column_rechecked.append('Exam_Date')
                exam_digit_points_user.extend(value)
                exam_digit_cordinates=exam_digit_points_user
                print("exam_digit_points_user ",exam_digit_points_user)
                ocr_field.append('Exam_Date')

            if key == 'Registration_OCR':
                reg_digit_points_user=[]
                #db_column_rechecked.append('Registration')
                reg_digit_points_user.extend(value)
                reg_digit_cordinates=reg_digit_points_user
                print("reg_digit_points_user ",reg_digit_points_user)
                ocr_field.append('Registration')
            
            
            if key == 'Subject_Code_OCR':
                
                subject_digit_points_user=[]
                #db_column_rechecked.append('Subject_Code')
                subject_digit_points_user.extend(value)
                subject_digit_cordinates=subject_digit_points_user
                print("subject_digit_points_user ,subject_digit_points_user")
                ocr_field.append('Subject_Code')
        
            
            # elif key != 'Roll_No_OCR':
            #     QMessageBox.information(self, "Error", f" Roll Number OCR is not found in cordinates file, please add Roll_No_OCR " )
            #     self.refresh_omr()
          
        update_global_variables(cleaned_data)
    
        self.columns_platform.setText(', '.join(column_field))    
        self.columns_platform_question.setText(', '.join(ques_field))
        self.columns_platform_ocr.setText(', '.join(ocr_field))
        del column_field
        del ques_field
       

       
    def btnstate(self,b):
        global Resize_CheckBox 
  
        if b.isChecked() == True:
            print (b.text()+" is selected")
            Resize_CheckBox ='True'
        else:
            Resize_CheckBox ='False'
            print (b.text()+" is deselected")      

def first_gui():
    app = QApplication(sys.argv)
    window = ScriptLauncher()
    window.show()
    app.exec_()

if __name__ == "__main__":
    mutex = ctypes.windll.kernel32.CreateMutexW(None, False, "Global\\BarcodeReaderMutex")
    last_error = ctypes.windll.kernel32.GetLastError()
    
    if last_error == 183:  # ERROR_ALREADY_EXISTS
        print("Another instance is already running.")
        sys.exit(1)
    
    first_gui()


