import web
import app

class SaveCode:

    def __init__(self):
        print("Save code")
        pass

    def create(self,code):
        file=open('static/code/code.py','w')
        for element in code:
            file.write(element+"\n")
        file.close()
        print("Create code.py")

    def append(self, code):
        file=open('static/code/code.py','a+')
        for element in code:
            file.write(element+"\n")
        file.close()
        print("Append code.py")
