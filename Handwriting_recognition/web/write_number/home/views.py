# -*- coding: utf-8 -*-
from django.shortcuts import render,reverse,redirect
from django.http import JsonResponse
import json
import os
import requests
import datetime
from write_number import settings
from .models import *
# Create your views here.
def index(request):
    if request.method=='GET':
        return render(request,"index.html")
    if request.method=='POST':
        up_file=request.FILES.get("file")
        img_url=request.POST.get("url")
        dt=datetime.datetime.now()
        file_name=dt.strftime("%Y%m%d")
        path=""
        # print(up_file)
        demo = file_dir()
        if img_url==None:
            demo.path=up_file
            demo.save()
        else:
            img_content=requests.get(img_url)
            path=os.path.join(settings.BASE_DIR,"static",'upload',file_name,"1.jpg")
            with open(path,"wb") as tf:
                tf.write(img_content.content)

        return JsonResponse({"flag":"ok"})

