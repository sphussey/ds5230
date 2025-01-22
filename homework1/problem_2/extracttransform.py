# homework 1 problem 2 - kosarak dataset
# this program downloads the text sparse dataset from the internet
# and loads it into a sparse matrix in arff format to be loaded
# into weka explorer
#
# RUNTIME: this script's runtime ~ 10-20 seconds
#
# WEKA UPLOAD TIME: loaded into Weka Explorer in under 5 seconds
#
#

import requests


def getdatafromlink():
    url = 'http://fimi.uantwerpen.be/data/kosarak.dat'
    r = requests.get(url)
    data = [sorted(set(map(int, line.split()))) for line in r.text.split('\n')]
    max_value = max(set(map(int, r.text.split())))
    return data, max_value

def converttoarff(data, max_value, filename):
    with open(filename, 'w') as arff:
        # write header with dataset name
        arff.write('@RELATION kosarak\n')
        # write attributes
        for attr in range(1, max_value + 1):
            arff.write("@ATTRIBUTE " + str(attr) + " {0, 1}\n")
        #write data
        arff.write('\n@DATA\n')
        for dataentry in data:
            if dataentry:
                dataentry = [entry - 1 for entry in dataentry]
                arff.write( '{ ' + ' 1, '.join(map(str, dataentry)) + ' 1 }\n')
            


def main():
    kosarak_raw, max_value = getdatafromlink()
    print(max_value)
    converttoarff(kosarak_raw, max_value, "problem_2/data/kosarak.arff")


if __name__ == '__main__':
    main()





