import os

def textbox_code():
    l = '<input type=\"text\"> '
    return l


def label_code():
    l = 'Label '
    return l


def radiobutton_code():
    l = '<input type=\"radio\" name=\"n\" value=\"v\"> '
    return l


def checkbox_code():
    l = '<input type=\"checkbox\"> '
    return l


def button_code():
    l = '<input type=\"submit\" value=\"Button\"> '
    return l


def image_code():
    #specify the path where the dummy image is persent, if not in the same dir
    l = '<img src=\"dummy_image.png\" height=50px alt=\"Image\"> '
    return l


def break_code():
    l = '\n<br><br>\n'
    return l

def generate_html():
    #path for storing the html file
    path='PATH_TO_DIR_FOR_STORING_HTML_FILE/'
    #define the opening tags of html 
    html=open(path+'generated_code.html','w')
    html.write('<HTML>\n<HEAD>\n')
    #link the css file
    #specify the full path in href if css not present in the same dir
    html.write('<link rel="stylesheet" type="text/css" href="stylesheet.css">\n')
    html.write('<TITLE>Generated HTML Code</TITLE>\n')
    html.write('</HEAD>\n<BODY>\n')
    
    temp = open(path+'temp.txt')
    lines=temp.readlines()
    for line in lines:
        #split the line to obtain element_name, x and y
        l=line.split(',')
        #l=['element_name', x, y]
        #generate code for the specific element
        if(l[0]=='TextBox'):
            c=textbox_code()
        elif(l[0]=='Label'):
            c=label_code()
        elif(l[0]=='RadioButton'):
            c=radiobutton_code()
        elif(l[0]=='CheckBox'):
            c=checkbox_code()
        elif(l[0]=='Button'):
            c=button_code()
        elif(l[0]=='Image'):
            c=image_code()
        elif(l[0]=='BREAK\n'):
            c=break_code()
        else:
            print('printing else code', l)
        html.write(c)
    
    html.write('\n</BODY>\n</HTML>')
    
    #html script is generated
    #destroy the temp file now
    #change the path to temp file, if not in the same dir
    os.remove(path+'temp.txt')

#generate_html()
