# DeepCWAVE

### To produce predictions
- Call `python RunModel.py` specifying the input, which can be an absolute or relative path to a netcdf file (passing entire directories here is currently unstable, working on fixing that)
- Optionally can specify the directory where the output file will be saved with the output flag, otherwise it will just save in the current working directory
- Don't use the weights argument yet, there's currently only one model anyway
- Call format: `python RunModel.py [-h] [--outdir OUTDIR] [--weights WEIGHTS] input`
- Calling `python RunModel.py -h` will give you more information about the arguments
- Sample call: `python RunModel.py /path/to/data/S1A_ALT_coloc201701S.nc -outdir /path/to/destination/`
  - This will read `/path/to/data/S1A_ALT_coloc201701S.nc` and produce an output file at `/path/to/destination/S1A_ALT_coloc201701S_preds.csv`

## Dependencies
- Python 3.6.8
- 'numpy': '1.16.2',
- 'sklearn': '0.19.2',
- 'pandas': '0.23.4',
- 'keras': '2.2.4',
- 'tensorflow': '1.11.0',
- 'netCDF4': '1.4.2'