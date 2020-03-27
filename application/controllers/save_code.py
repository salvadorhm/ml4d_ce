import web
import webdataminingtool

render = web.template.render('application/views/main', base="../master")

class SaveCode:

    def __init__(self):
        print("Save code")
        pass

    def create(self,code):
        try:
            file=open('static/code/code.py','w')
            for element in code:
                file.write(element+"\n")
            file.close()
            print("Create code.py")
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

    def append(self, code):
        try:
            file=open('static/code/code.py','a+')
            for element in code:
                file.write(element+"\n")
            file.close()
            print("Append code.py")
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

    def createCode(self,file, code):
        try:
            file=open('static/code/'+file,'w')
            for element in code:
                file.write(element+"\n")
            file.close()
            print("Append code.py")
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])
