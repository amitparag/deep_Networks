from prettytable import PrettyTable

def plotTable(iterations):

    t = PrettyTable(['Warmstarted', 'Coldstarted'])
    for i in iterations:
           t.add_row([i[0], i[1]])
    print(t)      
