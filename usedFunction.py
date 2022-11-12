"""
@author: Aberraouf MAOUDJ <abma@mmmi.sdu.dk>, Anders Lyhne Christensen <andc@mmmi.sdu.dk>
this function is used to have tracebility of our execution process
"""
import numpy as np
import decimal
import datetime

LOG_FILE = 'Q_tables/log.txt'

def log_and_display(msg):
    stat_file = open(LOG_FILE, 'a')
    #content = "[{0}] {1}\n".format(datetime.datetime.now(), msg)
    stat_file.write(msg+"\n")
    stat_file.close()
    print(msg, end="") # contenant
