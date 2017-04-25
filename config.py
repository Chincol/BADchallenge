'''
SUMMARY:  config file
AUTHOR:   Qiuqiang Kong
Created:  2016.10.15
Modified: 
--------------------------------------
'''

# warblrb dataset
wbl_root_fd = "/vol/vssp/msos/qk/warblrb10k_public_wav"
wbl_wav_fd = wbl_root_fd + "/wav"
wbl_csv_path = wbl_root_fd + "/warblrb10k_public_metadata.csv"

# freefield dataset
ff_root_fd = "/vol/vssp/msos/qk/ff1010_bird"
ff_wav_fd = ff_root_fd + "/wav"
ff_csv_path = ff_root_fd + "/ff1010bird_metadata.csv"

# your workspace
scrap_fd = "/vol/vssp/AcousticEventsDetection/bird_song/bird_backup_scrap"     # you need modify this path
mel_fd = "/vol/vssp/cvpwrkspc01/scratch/is0017"

# wbl dataset workspace
wbl_cv10_csv_path = scrap_fd + "/warblrb_cv10.csv"
wbl_denoise_wav_fd = scrap_fd + '/wbl_denoise_wav'
wbl_denoise_fe_fd = scrap_fd + '/wbl_denoise_fe'
wbl_denoise_fe_mel_fd = wbl_denoise_fe_fd + '/wbl_denoise_fe_mel'
wbl_denoise_fe_fft_fd = wbl_denoise_fe_fd + '/wbl_denoise_fe_fft'
wbl_dev_md_fd = scrap_fd + "/wbl_dev_md"
wbl_mel_fd = mel_fd + '/wbl_fe_mel'
wbl_fft_fd = scrap_fd + '/wbl_fe_fft'

# ff dataset workspace
ff_cv10_csv_path = scrap_fd + "/ff_cv10.csv"
ff_denoise_wav_fd = scrap_fd + '/ff_denoise_wav'
ff_denoise_fe_fd = scrap_fd + '/ff_denoise_fe'
ff_denoise_fe_mel_fd = ff_denoise_fe_fd + '/ff_denoise_fe_mel'
ff_denoise_fe_fft_fd = ff_denoise_fe_fd + '/ff_denoise_fe_fft'
ff_dev_md_fd = scrap_fd + "/ff_dev_md"
ff_mel_fd = mel_fd + '/ff_fe_mel'
ff_fft_fd = scrap_fd + '/ff_fe_fft'

# test dataset workspace
test_wav_fd = "/vol/vssp/msos/qk/test_bird_wav"
test_mel_fd = mel_fd + '/test_fe_mel'

# global params
win = 1024
fs = 44100.
n_duration = 440    # 44 frames per second, all together 10 seconds