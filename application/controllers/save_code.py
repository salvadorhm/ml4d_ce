import web
import ml4d

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

    def createCols(self,cols):
        try:
            file=open('static/deploy/cols.txt','w')
            for col in cols:
                file.write(col+"\n")
            file.close()
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

    def createModel(self,model):
        try:
            file=open('static/deploy/model.txt','w')
            file.write(model)
            file.close()
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

    def readCols(self):
        try:
            cols = []
            file = open('static/deploy/cols.txt','r')
            for line in file:
                cols.append(line.strip('\n'))
            return cols
        except Exception as e:
            print(e.args)
            return None

    def readModel(self):
        try:
            file = open('static/deploy/model.txt','r')
            return file.read()
        except Exception as e:
            print(e.args)
            return None