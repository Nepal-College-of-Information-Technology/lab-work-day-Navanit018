import os

class AppLauncher:
    def open_application(self, app_name):
        os.system(f'start {app_name}')  

        
