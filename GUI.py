from tkinter.filedialog import askopenfilename
import os
import tkinter as tk
from tkinter import *  # 图形界面库
import tkinter.messagebox as messagebox  # 弹窗
from ttkbootstrap import Style
from PIL import Image, ImageTk
import threading
import tkinter.font as tf
import numpy as np
from Filters import LMS_AdaFilter, RLS_AdaFilter, noise_generator,NLMS_AdaFilter
from Application import getSNR, getMSE, getFeature
import matplotlib.pyplot as plt
import scipy.signal as sig


class AdapFilterGUI:
    def __init__(self,root):
        self.initGUI(root)


    def initGUI(self,root):
        self.style = Style(theme='lumen')
        self.window = root
        self.window = self.style.master
        self.window.title('AdapFilter')
        self.window.geometry('1280x750+120+25')
        self.window.iconphoto(True, tk.PhotoImage(file='tt1.png'))
        self.data_path = StringVar()
        self.data_path.set(os.path.abspath("."))
        ft = tf.Font(family='微软雅黑', size=11)
        self.fm = tk.LabelFrame(self.window, text='主面板',font=ft, width=1200, height=720)
        self.left = tk.Frame(self.fm, bg='white', width=400, height=720)
        self.data_input = tk.LabelFrame(self.left, text='数据载入',font=ft, width=400, height=90)
        self.nos_setting = tk.LabelFrame(self.left, text='非平稳噪声设置',font=ft, width=400, height=90)
        self.flt_setting = tk.LabelFrame(self.left, text='滤波器设置',font=ft, width=400, height=220)
        self.eval = tk.LabelFrame(self.left, text='效果评价',font=ft, width=400, height=200)
        self.feature = tk.LabelFrame(self.left, text='R波分析',font=ft, width=400, height=150)

        self.right = tk.Frame(self.fm, bg='white', width=800, height=720)
        self.org_show = tk.LabelFrame(self.right, text='原始信号',font=ft, width=800, height=220)
        self.nos_show = tk.LabelFrame(self.right, text='加噪信号',font=ft, width=800, height=220)
        self.flt_show = tk.LabelFrame(self.right, text='滤波信号',font=ft, width=800, height=220)

        self.fm.grid(row=0, column=0, padx=15, pady=10,ipadx=3,ipady=3)
        self.left.grid(row=0, column=0, padx=10, pady=3)
        self.right.grid(row=0, column=1, padx=10, pady=3)
        self.data_input.grid(row=0, column=0, padx=0, pady=6)
        self.nos_setting.grid(row=1, column=0, padx=0, pady=6)
        self.flt_setting.grid(row=2, column=0, padx=0, pady=6)
        self.eval.grid(row=3, column=0, padx=0, pady=6)
        self.feature.grid(row=4, column=0, padx=0, pady=6)
        self.org_show.grid(row=0, column=0, padx=5, pady=2)
        self.org_show.pack_propagate(0)
        self.nos_show.grid(row=1, column=0, padx=5, pady=2)
        self.nos_show.pack_propagate(0)
        self.flt_show.grid(row=2, column=0, padx=5, pady=2)
        self.flt_show.pack_propagate(0)

        self.showLen = 1000
        self.ok = 0
        self.f1 = None
        self.f2 = None
        self.f3 = None

        #信号输入
        Label(self.data_input, text="选择路径:  ",font=ft).grid(row=0, column=0, padx=3, pady=2)
        Entry(self.data_input, textvariable=self.data_path,width=27,font=ft).grid(row=0, column=1, padx=2, pady=2)
        Button(self.data_input, text="打开文件",width=33,bd=2,relief='groove',font=ft, command=lambda:self.selectData()).grid(row=1, column=0,columnspan=2, padx=0, pady=5)

        #噪声设置
        self.snr = StringVar(value='10')
        Label(self.nos_setting, text="信噪比:    ",font=ft).grid(row=1, column=0, padx=20, pady=2)
        Label(self.nos_setting, text="dB",font=ft).grid(row=1, column=2,  padx=0, pady=2)
        Entry(self.nos_setting, textvariable=self.snr, width=8,font=ft).grid(row=1, column=1,padx=0, pady=2)
        Button(self.nos_setting, text="确定",bd=2,font=ft,relief='groove',width=8, command=lambda: self.getNoise()).grid(row=1, column=3,padx=20, pady=2)

        #滤波器设置
        Label(self.flt_setting, text="自适应算法选择", font=ft, width=18).grid(row=0, column=0, padx=0, pady=2)
        Label(self.flt_setting, text="参数设置", font=ft, width=18).grid(row=0, column=1,columnspan=2, padx=0, pady=2)
        v = tk.IntVar()
        af1 = tk.Radiobutton(self.flt_setting, text='LMS', font=ft, variable=v, value=1)
        af2 = tk.Radiobutton(self.flt_setting, text='NLMS', font=ft, variable=v, value=2)
        af3 = tk.Radiobutton(self.flt_setting, text='RLS', font=ft, variable=v, value=3)
        af1.grid(row=1, column=0, padx=0, pady=2)
        af2.grid(row=2, column=0, padx=0, pady=2)
        af3.grid(row=3, column=0, padx=0, pady=2)

        self.mu = StringVar(value='0.05')
        self.N = StringVar(value='4')
        Label(self.flt_setting, text="步长因子:", font=ft).grid(row=1, column=1, padx=0, pady=2)
        Label(self.flt_setting, text="滑窗大小:", font=ft).grid(row=2, column=1, padx=0, pady=2)
        Entry(self.flt_setting, textvariable=self.mu, width=8,font=ft).grid(row=1, column=2, padx=0, pady=2)
        Entry(self.flt_setting, textvariable=self.N, width=8,font=ft).grid(row=2, column=2, padx=0, pady=2)
        Button(self.flt_setting, text="确定", width=12,bd=2,font=ft,relief='groove',command=lambda: self.getFlt(v.get())).grid(row=3, column=1,columnspan=2, padx=0,pady=2)


        #评价
        self.snr0 = tk.StringVar()
        self.snr1 = tk.StringVar()
        self.mse0 = tk.StringVar()
        self.mse1 = tk.StringVar()
        Label(self.eval, text="滤波前", font=ft).grid(row=1, column=1, padx=0, pady=2)
        Label(self.eval, text="滤波后", font=ft).grid(row=1, column=2, padx=0, pady=2)
        Label(self.eval, text="SNR", font=ft).grid(row=2, column=0, padx=0, pady=2)
        Label(self.eval, text="MSE", font=ft).grid(row=3, column=0, padx=0, pady=2)
        self.old_snr=Label(self.eval, width=10,textvariable=self.snr0,bd=1,relief='groove')
        self.old_snr.grid(row=2, column=1, padx=0, pady=2)
        self.new_snr = Label(self.eval, width=10,textvariable=self.snr1,bd=1,relief='groove')
        self.new_snr.grid(row=2, column=2, padx=0, pady=2)

        self.old_mse = Label(self.eval, width=10,textvariable=self.mse0,bd=1,relief='groove')
        self.old_mse.grid(row=3, column=1, padx=0, pady=2)
        self.new_mse = Label(self.eval, width=10,textvariable=self.mse1,bd=1,relief='groove')
        self.new_mse.grid(row=3, column=2, padx=0, pady=2)
        Button(self.eval, text="分 析", width=35, bd=2, font=ft, relief='groove',command=lambda: self.showEval()).grid(row=4, column=0,columnspan=3, padx=5,pady=2)

       #特征提取
        self.hr = tk.StringVar()
        self.rr = tk.StringVar()
        Label(self.feature, text="心率", font=ft).grid(row=1, column=0, padx=0, pady=2)
        Label(self.feature, text="R-R间期", font=ft).grid(row=2, column=0, padx=0, pady=2)
        Label(self.feature, text="次/分", font=ft).grid(row=1, column=2, padx=0, pady=2)
        Label(self.feature, text="s", font=ft).grid(row=2, column=2, padx=0, pady=2)
        self.hrshow = Label(self.feature, width=10,textvariable=self.hr,bd=1,relief='groove')
        self.hrshow.grid(row=1, column=1, padx=0, pady=2)
        self.rrshow = Label(self.feature, width=10,textvariable=self.rr,bd=1,relief='groove')
        self.rrshow.grid(row=2, column=1, padx=0, pady=2)
        Button(self.feature, text="分 析", width=35, bd=2, font=ft, relief='groove', command=lambda: self.showFeat()).grid(row=3, column=0, columnspan=3, padx=5, pady=2)

        self.window.mainloop()  # 主消息循环

    #选择输入信号数据路径
    def selectData(self):
        path_ = askopenfilename()  # 使用askdirectory()方法返回文件夹的路径
        if path_ == "":
            self.data_path.get()  # 当打开文件路径选择框后点击"取消" 输入框会清空路径，所以使用get()方法再获取一次路径
        else:
            path_ = path_.replace("/", "\\")  # 实际在代码中执行的路径为“\“ 所以替换一下
            self.data_path.set(path_)
            if '.npy' in path_:
                self.getData()
            else:
                messagebox.showinfo('错误！', '文件格式出错')

    #评价指标计算并显示
    def showEval(self):
        if self.ok == 2:
            snr0 = getSNR(self.org,self.noise_data)
            snr1 = getSNR(self.org[:len(self.flt_data)],self.flt_data)
            self.snr0.set(str(round(snr0, 2)))
            self.snr1.set(str(round(snr1, 2)))

            mse0 = getMSE(self.org,self.noise_data) * 1000
            mse1 = getMSE(self.org[:len(self.flt_data)],self.flt_data) * 1000
            self.mse0.set(str(round(mse0, 2)))
            self.mse1.set(str(round(mse1, 2)))
        else:
            messagebox.showinfo('错误！', '请重新操作')

    #R波分析结果计算并显示
    def showFeat(self):
        if self.ok == 2:
            rr,hr = getFeature(self.flt_data)
            self.hr.set(str(round(hr, 1)))
            self.rr.set(str(round(rr, 2)))

            data_mean = np.mean(self.flt_data[-self.showLen:])
            p_id, _ = sig.find_peaks(self.flt_data[-self.showLen:], distance=70, height=data_mean*1.1)
            p_val = self.flt_data[-self.showLen:][p_id]  # 取出峰值对应的幅值
            global rPic
            plt.figure()
            plt.plot(p_id, p_val, 'ro')
            plt.plot(self.flt_data[-self.showLen:])
            plt.axis('off')
            plt.savefig("r.png", bbox_inches='tight')
            img4 = Image.open('r.png')
            img4 = img4.resize((800, 220))
            rPic = ImageTk.PhotoImage(img4)
            self.f3.configure(image = rPic )
            self.f3.image = rPic

        else:
            messagebox.showinfo('错误！', '请重新操作')



    #读取数据
    def getData(self):
        self.org = np.load(self.data_path.get())[:-100]
        self.show(1)
        self.ok = 1

    #获取加噪信号
    def getNoise(self):
        if self.ok == 1:
            self.noise = noise_generator(self.org, float(self.snr.get()))
            self.noise_data = self.org + self.noise
            self.show(2)
            self.ok = 2
        else:
            messagebox.showinfo('错误！', '请先加载原数据')

    #进行滤波
    def getFlt(self,flag):
        if self.ok == 2:
            if flag == 1:
                self.flt_data =  LMS_AdaFilter(self.noise, self.noise_data,int(self.N.get()),float(self.mu.get()))
                self.show(3)
            elif flag == 2:
                self.flt_data = NLMS_AdaFilter(self.noise, self.noise_data,int(self.N.get()),float(self.mu.get()))
                self.show(3)
            elif flag == 3:
                self.flt_data = RLS_AdaFilter(self.noise, self.noise_data,int(self.N.get()))
                self.show(3)
            else:
                messagebox.showinfo('错误！', '请选择一个算法')
        elif self.ok == 0:
            messagebox.showinfo('错误！', '请先加载原数据')
        else:
            messagebox.showinfo('错误！', '请先加载噪声数据')

    # 显示信号
    def show(self,flag):
        if flag == 1:

            global orgPic
            plt.figure()
            plt.plot(self.org[-self.showLen:])
            plt.axis('off')
            plt.savefig("org.png", bbox_inches='tight')
            img1 = Image.open('org.png')
            img1 = img1.resize((800, 220))
            orgPic = ImageTk.PhotoImage(img1)
            if not self.f1:
                self.f1 = Label(self.org_show, image=orgPic)
                self.f1.pack()
            else:
                self.f1.configure(image=orgPic)
                self.f1.image = orgPic
        elif flag == 2:
            global noisePic
            plt.figure()
            plt.plot(self.noise_data[-self.showLen:])
            plt.axis('off')
            plt.savefig("noise.png", bbox_inches='tight')
            img2 = Image.open('noise.png')
            img2 = img2.resize((800, 220))
            noisePic = ImageTk.PhotoImage(img2)
            if not self.f2:
                self.f2 = Label(self.nos_show, image=noisePic)
                self.f2.pack()
            else:
                self.f2.configure(image=noisePic)
                self.f2.image = noisePic
        elif flag == 3:
            global fltPic
            plt.figure()
            plt.plot(self.flt_data[-self.showLen:])
            plt.axis('off')
            plt.savefig("flt.png", bbox_inches='tight')
            img3 = Image.open('flt.png')
            img3 = img3.resize((800, 220))
            fltPic = ImageTk.PhotoImage(img3)
            if not self.f3:
                self.f3 = Label(self.flt_show, image=fltPic)
                self.f3.pack()
            else:
                self.f3.configure(image=fltPic)
                self.f3.image = fltPic




if __name__ == '__main__':
    try:
        root = Tk()
        AdapFilterGUI(root)
    except:
        messagebox.showinfo('错误！', '请重新操作')