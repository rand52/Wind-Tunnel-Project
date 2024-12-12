import twoD_Datapoint as dp

# datafile
datafile = "raw_2D.txt"
comment_indicator = "#" # line indicator NOT read
header = 2 # header NOT read size

# list with the datapoints
datapoints = []

with open(datafile, 'r') as file:
    l=0
    for line in file:
        l+=1
        if l<=header: continue # skip header
        line = line.strip()
        if not line.startswith(comment_indicator): # skip commented lines
            datPt = dp.twoD_DP() # create new datapoint
            e=0
            for entry in line.split():
                e+=1
                if e>2: entry_val=float(entry)
                # save the datapoint entries in the correct fields
                match e:
                    case 3 : datPt.aoa=entry_val
                    case 4 : datPt.del_pb=entry_val
                    case 5 : datPt.p_atm=entry_val*100 # it's given in kPa
                    case 6 : datPt.temp_C=entry_val
                    case 8 : datPt.rho=entry_val
                    case 105: datPt.p_stat=entry_val # P097
                    case x if (x >= 9 and x <= 33): datPt.airfoil_top_p_taps.append(entry_val) # P001-P025
                    case x if (x >= 34 and x <= 57): datPt.airfoil_bottom_p_taps.append(entry_val) # P026-P049
                    case x if (x >= 58 and x <= 104): datPt.rake_total_p_taps.append(entry_val) # P050-P096
                    case x if (x >= 106 and x <= 117): datPt.rake_total_p_taps.append(entry_val) # P098-P109
            datapoints.append(datPt)

