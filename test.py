import configparser

config = configparser.ConfigParser()
config["Default"] = {'Data':'../Data/iris.csv'}
with open('config.ini','w') as configfile:
    config.write(configfile)