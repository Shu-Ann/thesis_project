import pandas as pd
from pydub import AudioSegment

# read files
pepper_1=pd.read_csv('./data/processed/pepper_1_merge.csv')
pepper_2=pd.read_csv('./data/processed/pepper_2_merge.csv')
pepper_3=pd.read_csv('./data/processed/pepper_3_merge.csv')
pepper_4=pd.read_csv('./data/processed/pepper_4_merge.csv')

fetch_1=pd.read_csv('./data/processed/fetch_1_merge.csv')
fetch_2=pd.read_csv('./data/processed/fetch_2_merge.csv')
fetch_3=pd.read_csv('./data/processed/fetch_3_merge.csv')
fetch_4=pd.read_csv('./data/processed/fetch_4_merge.csv')

audio_p1 = AudioSegment.from_file('./data/audio/pepper_1.m4a')
audio_p2 = AudioSegment.from_file('./data/audio/pepper_2.m4a')
audio_p3 = AudioSegment.from_file('./data/audio/pepper_3.m4a')
audio_p4 = AudioSegment.from_file('./data/audio/pepper_4.m4a')

audio_f1 = AudioSegment.from_file('./data/audio/fetch_1.m4a')
audio_f2 = AudioSegment.from_file('./data/audio/fetch_2.m4a')
audio_f3 = AudioSegment.from_file('./data/audio/fetch_3.m4a')
audio_f4 = AudioSegment.from_file('./data/audio/fetch_4.m4a')

csv_name=['pepper_1','pepper_2', 'pepper_3', 'pepper_4','fetch_1','fetch_2', 'fetch_3','fetch_4']

csv_list=[pepper_1, pepper_2, pepper_3, pepper_4, fetch_1 , fetch_2, fetch_3, fetch_4]

audio_list=[audio_p1, audio_p2,audio_p3, audio_p4, audio_f1, audio_f2, audio_f3, audio_f4]

# ------ segmentation -------

for c in range(0, len(csv_list)):
  start=csv_list[c]['Start'].tolist()
  end=csv_list[c]['End'].tolist()
  csv_list[c]['file_name']='0'
  audio=audio_list[c]
  for i in range(len(start)):
    chunk_data = audio[start[i]*1000:end[i]*1000]
    if csv_list[c]['Role'][i]=='R':
          chunk_data.export('./data/audio/seg_R/'+csv_name[c]+'_R_audio'+str(i)+'.wav', format="wav")
          csv_list[c]['file_name'][i]=csv_name[c]+'_R_audio'+str(i)+'.wav'
    elif csv_list[c]['Role'][i]=='P':
          chunk_data.export('./data/audio/seg_P/'+csv_name[c]+'_P_audio'+str(i)+'.wav', format="wav")
          csv_list[c]['file_name'][i]=csv_name[c]+'_P_audio'+str(i)+'.wav'

pepper_1_R=pepper_1[pepper_1['Role']=='R']
pepper_1_P=pepper_1[pepper_1['Role']=='P']

pepper_2_R=pepper_2[pepper_2['Role']=='R']
pepper_2_P=pepper_2[pepper_2['Role']=='P']

pepper_3_R=pepper_3[pepper_3['Role']=='R']
pepper_3_P=pepper_3[pepper_3['Role']=='P']

pepper_4_R=pepper_4[pepper_4['Role']=='R']
pepper_4_P=pepper_4[pepper_4['Role']=='P']

fetch_1_R=fetch_1[fetch_1['Role']=='R']
fetch_1_P=fetch_1[fetch_1['Role']=='P']

fetch_2_R=fetch_2[fetch_2['Role']=='R']
fetch_2_P=fetch_2[fetch_2['Role']=='P']

fetch_3_R=fetch_3[fetch_3['Role']=='R']
fetch_3_P=fetch_3[fetch_3['Role']=='P']

fetch_4_R=fetch_4[fetch_4['Role']=='R']
fetch_4_P=fetch_4[fetch_4['Role']=='P']

pepper_1_R.to_csv('./data/audio/pepper_1_R_audio.csv')
pepper_2_R.to_csv('./data/audio/pepper_2_R_audio.csv')
pepper_3_R.to_csv('./data/audio/pepper_3_R_audio.csv')
pepper_4_R.to_csv('./data/audio/pepper_4_R_audio.csv')

fetch_1_R.to_csv('./data/audio/fetch_1_R_audio.csv')
fetch_2_R.to_csv('./data/audio/fetch_2_R_audio.csv')
fetch_3_R.to_csv('./data/audio/fetch_3_R_audio.csv')
fetch_4_R.to_csv('./data/audio/fetch_4_R_audio.csv')

pepper_1_P.to_csv('./data/audio/pepper_1_P_audio.csv')
pepper_2_P.to_csv('./data/audio/pepper_2_P_audio.csv')
pepper_3_P.to_csv('./data/audio/pepper_3_P_audio.csv')
pepper_4_P.to_csv('./data/audio/pepper_4_P_audio.csv')

fetch_1_P.to_csv('./data/audio/fetch_1_P_audio.csv')
fetch_2_P.to_csv('./data/audio/fetch_2_P_audio.csv')
fetch_3_P.to_csv('./data/audio/fetch_3_P_audio.csv')
fetch_4_P.to_csv('./data/audio/fetch_4_P_audio.csv')

          
