from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('', views.home, name='home'),  # 主页路由
    path('system/', views.system, name='system'),  # 系统页面路由
    path('about_us/', views.about_us, name='about_us'),  # 关于我们页面路由
    path('save_photo/', views.save_photo, name='save_photo'),
    path('process_and_train/', views.process_and_train, name='process_and_train'),
    path('process_frame/', views.process_frame, name='process_frame'),

    #path('upload/', views.upload_file, name='upload_file'),  # 文件上传处理路由
    #path('detection_result/', views.detection_result, name='detection_result'),  # 检测结果页面路由（可选）
]
