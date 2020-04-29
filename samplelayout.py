#IMPORTING MODULES
from tkinter import*
import tkinter as tk
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import csv

#SETTING UP PRIMARY WINDOW
window = Tk()
window.geometry("1366x768")
window.title("Heart Disease Predictor")
#window.configure(background='ivory3')

#FETCHING DATA AND PASSING TO TRAINED MODEL
def add():
    model=joblib.load("TrainedModel(LoR).pkl")
    col1=int(c1.get())
    col2=int(c2.get())
    col3=int(c3.get())
    col4=int(c4.get())
    col5=int(c5.get())
    col6=int(c6.get())
    col7=int(c7.get())
    col8=int(c8.get())
    col9=int(c9.get())
    col10=float(c10.get())
    col11=int(c11.get())
    col12=int(c12.get())
    col13=int(c13.get())

    Xnew=[[col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13]]
    col14=model.predict(Xnew)
    Xcsv=[[col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14]]
    with open('RecordHD.csv', 'a',newline='') as f:
        writer = csv.writer(f)
        writer.writerows(Xcsv)
    
    if(model.predict(Xnew)==1):
        Label(genframe,text="Heart Disease NEGATIVE", font="Calibri 20",bg="whitesmoke",fg="green").place(x=540,y=625)
    else:
        Label(genframe,text="Heart Disease POSITIVE", font="Calibri 20",bg="whitesmoke",fg="red").place(x=540,y=625)

def acc():
    Label(mainframe,text="97.7777%", font="Calibri 15",bg="whitesmoke").place(x=60,y=230)

def cm():
    Label(mainframe,text="   1   2\n1[51  1]\n2[1  37]", font="Calibri 15",bg="whitesmoke").place(x=60,y=370)
        
    

    
      
     
###########GUI############
        
genframe=Frame(window,width=820, height=500,bg="whitesmoke").place(x=250,y=100)
header=Frame(window,width=820, height=60,bg="steelblue").place(x=250,y=100)

fname=Label(genframe,text='Health Details', font="Calibri 20 bold",bg="steelblue",fg="white").place(x=590,y=115)
#INPUTS:
#Age
c1 = StringVar()
Label(genframe,text='Age', font="Calibri 15",bg="whitesmoke").place(x=300,y=180)
Entry(genframe,width=30,textvariable=c1).place(x=420,y=185)

#Sex
c2 = StringVar()
Label(genframe,text='Sex(M:1,F:0)', font="Calibri 15",bg="whitesmoke").place(x=300,y=220)
Entry(genframe,width=30,textvariable=c2).place(x=420,y=225)

#Chest Pain Type
c3 = StringVar()
Label(genframe,text='CPT(1-4)', font="Calibri 15",bg="whitesmoke").place(x=300,y=260)
Entry(genframe,width=30,textvariable=c3).place(x=420,y=265)

#Resting Blood Pressure
c4 = StringVar()
Label(genframe,text="RestingBP", font="Calibri 15",bg="whitesmoke").place(x=300,y=300)
Entry(genframe,width=30,textvariable=c4).place(x=420,y=305)

#Serum Cholestrol
c5=StringVar()
Label(genframe,text='SC(>100)', font="Calibri 15",bg="whitesmoke").place(x=300,y=340)
Entry(genframe,width=30,textvariable=c5).place(x=420,y=345)

#Fasting Blood Sugar
c6=StringVar()
Label(genframe,text='FBS(Y:1,N:0)', font="Calibri 15",bg="whitesmoke").place(x=300,y=380)
Entry(genframe,width=30,textvariable=c6).place(x=420,y=385)

#Resting ECG
c7=StringVar()
Label(genframe,text='R-ECG(0-2)', font="Calibri 15",bg="whitesmoke").place(x=300,y=420)
Entry(genframe,width=30,textvariable=c7).place(x=420,y=425)

#Maximum Heart Rate
c8=StringVar()
Label(genframe,text='MaxHR', font="Calibri 15",bg="whitesmoke").place(x=700,y=180)
Entry(genframe,width=30,textvariable=c8).place(x=820,y=185)

#Exercise Induced Angina
c9=StringVar()
Label(genframe,text='EIA(Y:1,N:0)', font="Calibri 15",bg="whitesmoke").place(x=700,y=220)
Entry(genframe,width=30,textvariable=c9).place(x=820,y=225)

#OldPeak
c10=StringVar()
Label(genframe,text='OP(0.0-5.0)', font="Calibri 15",bg="whitesmoke").place(x=700,y=260)
Entry(genframe,width=30,textvariable=c10).place(x=820,y=265)

#Slope of the peak
c11=StringVar()
Label(genframe,text='SOP(1-3)', font="Calibri 15",bg="whitesmoke").place(x=700,y=300)
Entry(genframe,width=30,textvariable=c11).place(x=820,y=305)

#Number of Major Vessels
c12=StringVar()
Label(genframe,text='NoMV(0-3)', font="Calibri 15",bg="whitesmoke").place(x=700,y=340)
Entry(genframe,width=30,textvariable=c12).place(x=820,y=345)

#Thal
c13=StringVar()
Label(genframe,text='Thal(3/6/7)', font="Calibri 15",bg="whitesmoke").place(x=700,y=380)
Entry(genframe,width=30,textvariable=c13).place(x=820,y=385)

Add=Button(genframe, text="Submit",width="15",font="Calibri 15 bold",command=add).place(x=745,y=505,anchor=E)


    
##Heading
titleframe=Frame(window, width=1366, height=60,bg="lightgrey").grid(column=0,row=0)
title=Label(titleframe,text="Prediction System(HDP)",font="Calibri 25 bold",bg="lightgrey",fg="steelblue").grid(column=0,row=0)
##Menu
logoframe=Frame(window,width=200, height=50,bg="lightskyblue").place(x=0,y=60)
menu=Label(logoframe,text="MODEL\nINFORMATION",font="Calibri 15 bold",bg="lightskyblue",fg="white").place(x=35,y=60)
##Main
mainframe=Frame(window,width=200, height=1366,bg="white").place(x=0,y=110)

#Buttons
#photo=PhotoImage(file="img")
Accuracy=Button(mainframe, text="Accuracy",width="15",height="2",font="Calibri 12",command=acc).place(x=160,y=200,anchor=E)
ConfusionMat=Button(mainframe, text="Confusion\nMatrix",width="15",height="2",font="Calibri 12",command=cm).place(x=160,y=340,anchor=E)
#Map=Button(mainframe, text="Option 3",width="15",height="2",font="Calibri 12").place(x=160,y=480,anchor=E)


#Help
helpFrame=Frame(window,width=200,height=1366,bg="whitesmoke").place(x=1160,y=60)
Label(helpFrame,text="Helping Desk", font="Calibri 20",bg="whitesmoke").place(x=1180,y=65)
Label(helpFrame,text="->Age-Enter patient's Age", font="Calibri 10",bg="whitesmoke").place(x=1170,y=100)
Label(helpFrame,text="->Sex-Enter patient's Gender", font="Calibri 10",bg="whitesmoke").place(x=1170,y=120)
Label(helpFrame,text="->CPT-Chest Pain Type(1/2/3/4)", font="Calibri 10",bg="whitesmoke").place(x=1170,y=140)
Label(helpFrame,text="->RestingBP-Resting Blood Pressure", font="Calibri 10",bg="whitesmoke").place(x=1170,y=160)
Label(helpFrame,text="->SC-Serum Cholestrol", font="Calibri 10",bg="whitesmoke").place(x=1170,y=180)
Label(helpFrame,text="->FBS-Fasting Blood Sugar", font="Calibri 10",bg="whitesmoke").place(x=1170,y=200)
Label(helpFrame,text="->RECG-Resting ECG", font="Calibri 10",bg="whitesmoke").place(x=1170,y=220)
Label(helpFrame,text="->MaxHR-Maximum Heart Rate", font="Calibri 10",bg="whitesmoke").place(x=1170,y=240)
Label(helpFrame,text="->EIA-Exercise Induced Agina", font="Calibri 10",bg="whitesmoke").place(x=1170,y=260)
Label(helpFrame,text="->OP-OldPeak", font="Calibri 10",bg="whitesmoke").place(x=1170,y=280)
Label(helpFrame,text="->SOP-Slope of Peak", font="Calibri 10",bg="whitesmoke").place(x=1170,y=300)
Label(helpFrame,text="->NoMV-Number of Major Vessels", font="Calibri 10",bg="whitesmoke").place(x=1170,y=320)
Label(helpFrame,text="->Thal-Thalassemia\n3.Normal\n6.Fixed defect\n7.Reversable defect", font="Calibri 10",bg="whitesmoke").place(x=1170,y=340)

window.mainloop()
