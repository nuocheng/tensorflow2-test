from django.db import models

# Create your models here.

class file_dir(models.Model):
    path=models.FileField(upload_to="%Y%m%d",null=False,blank=False)
    create_at=models.DateTimeField(auto_now_add=True)
    class Meta:
        db_table="file"
