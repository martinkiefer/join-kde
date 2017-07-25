import os

#Load individual 
d = '.'
dirs =  filter(lambda x: os.path.isdir(os.path.join(d, x)), os.listdir(d))

for dir in dirs:
    #Grab the common header
    f = open("./gnuplot.header","r")
    sh = f.read()
    f.close()

    f = open("./%s/gnuplot.custom" % dir,"r")
    sm = f.read()
    f.close()

    f = open("./%s/KernelGenerator/5_Postgres.results.group.results" % dir,"r")
    pv = f.read()
    f.close()

    #Grab the common footer
    f = open("./gnuplot.footer","r")
    sf = f.read()
    sf = sf % pv.split(",")[7].rstrip()
    f.close()

    f = open("./%s/plot.gnuplot" % dir,"w")
    f.write(sh+"\n"+sm+"\n"+sf)
    f.close()
