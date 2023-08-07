import numpy as np
from datetime import datetime

def dt5202_crunch(file_name, num_ev=100000000, num_ch=1):

    # define data types for header, event header, and event
    h_dt = np.dtype([('data_format', 'i2'), 
                     ('software_version', 'i1',3),
                     ('acq_mode', 'i1'),
                     ('time_unit','i1'),
                     ('energy_nChannels','i2'),
                     ('time_conversion','f4'),
                     ('start_acq', 'i8')
                    ])
    '''
    h_dt = np.dtype([('data_format', 'i2'), 
                     ('software_version', 'i1',3),
                     ('acq_mode', 'i1'),
                     ('start_acq', 'i8')
                    ])
   ''' 
    # event header data 
    eh_dt = np.dtype([('event_size', 'i2'), 
                     ('board_id', 'i1'),
                     ('trigger_ts', 'f8'),
                     ('trigger_id', 'i8'),
                     ('channel_mask', 'i8'),
                    ])
    # channel data structure
    ch_dt = np.dtype([('channel_id', 'i1'), 
                     ('data_type', 'i1'),
                     ('low_gain', 'i2'),
                     ('high_gain', 'i2'),
                    ])    

    # combining event header with 64 channels of data
    dt = np.dtype([('header',eh_dt), ('data',ch_dt,num_ch)])
    
    # read data
    with open(file_name, 'rb') as f:
        header = np.fromfile(f,dtype=h_dt,count=1)
        print (f"{header}")
        start_time = datetime.fromtimestamp(header['start_acq'][0]/1000)
        # print(f"Start Time: {start_time}")
        
        data = np.fromfile(f,dtype=dt,count=num_ev)
        
        data['header']['trigger_ts'] = data['header']['trigger_ts'].astype('float64')
        data['header']['trigger_ts'] /= 1e6
        data['header']['trigger_ts'] += header['start_acq'][0]/1000.
        print (f"{data[0]}")
        #print (f"{data[1]}")
        #print (f"{data}")
    return data



if __name__ == "__main__":
    import sys
    
    files = sys.argv[1:]
    
    for f in files:
        base_name = f[0:-4]
        print(f"Reading file: {f}")
        data = dt5202_crunch(f)
        np.save(base_name+'.npy', data)

