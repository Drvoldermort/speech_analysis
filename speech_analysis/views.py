from django.shortcuts import render_to_response
from django.http import HttpResponse
from django.views.decorators.csrf import ensure_csrf_cookie
from speech_analysis.settings import IMAGES_FOLDER
def get_tone_score(request):
    try:
        
        import io
        from watson_developer_cloud import ToneAnalyzerV3
        import json
        def toneanalyzer(data):
            tone_analyzer = ToneAnalyzerV3(
                version ='2017-09-21',
                username ='75302330-4d62-4e21-80b2-da038310d5d3',
                password ='Y85VmxzcFojj'
                )
            text = data
            content_type = 'application/json'
            tone = tone_analyzer.tone({"text": text},content_type)
            tone_data=(json.loads(json.dumps(tone, indent=2)))
            sentences_tone=tone_data["sentences_tone"]
            sentences=[]
            for sentence in sentences_tone:
                d={}
                d["sentence_no"]=int(sentence["sentence_id"])+1;
                d["sentence"]=sentence["text"];
                mood={}
                for tone_name in sentence["tones"]:
                    mood[tone_name["tone_name"]]=float(tone_name["score"])*100
                    d["mood"]=mood
                    
                sentences.append(d);
                           
            tones=tone_data["document_tone"]["tones"];
            
            dataPoints=[["Mood","Score"]]
            for tone in tones:
                dic=[]
                dic.append(tone["tone_name"]);
                dic.append(float(tone["score"])*100);
                dataPoints.append(dic);
            return dataPoints,sentences;

            
        def transcribe_file_with_auto_punctuation(path):
            """Transcribe the given audio file with auto punctuation enabled."""
            from google.cloud.speech import enums
            from google.cloud.speech import types
            from google.cloud import speech_v1p1beta1 as speech
            client = speech.SpeechClient()

            client = speech.SpeechClient()

            with io.open(path, 'rb') as audio_file:
                content = audio_file.read()

            audio = speech.types.RecognitionAudio(content=content)
            config = speech.types.RecognitionConfig(
                encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
                language_code='en-US',
                # Enable automatic punctuation
                enable_automatic_punctuation=True)

            operation = client.long_running_recognize(config, audio)
            response = operation.result(timeout=90)
            stt=""
            for result in response.results:
                stt+=result.alternatives[0].transcript;
            dataPoints,sentences=toneanalyzer(stt)
            return dataPoints,stt,sentences;   
        # [END speech_transcribe_file_with_auto_punctuation]

        def transcribe_gcs(gcs_uri):
            """Asynchronously transcribes the audio file specified by the gcs_uri."""
            #from google.cloud import speech
            from google.cloud.speech import enums
            from google.cloud.speech import types
            from google.cloud import speech_v1p1beta1 as speech

            client = speech.SpeechClient()

            audio = speech.types.RecognitionAudio(uri=gcs_uri)
            config = speech.types.RecognitionConfig(

                sample_rate_hertz=16000,
                language_code='en-US',
                enable_automatic_punctuation=True)

            operation = client.long_running_recognize(config, audio)

            print('Waiting for operation to complete...')
            response = operation.result(timeout=90)

            stt=""
            for result in response.results:
                stt+=result.alternatives[0].transcript;
            dataPoints,sentences=toneanalyzer(stt)
            return dataPoints,stt,sentences;

        def get_gender(pathy):
            import os
            import pickle as cPickle
            import numpy as np
            from scipy.io.wavfile import read
            import python_speech_features as mfcc
            from sklearn import preprocessing
            import warnings
        
            warnings.filterwarnings("ignore")
            def get_MFCC(sr,audio):
                features = mfcc.mfcc(audio,sr, 0.025, 0.01, 13,appendEnergy = False)
                feat     = np.asarray(())
                for i in range(features.shape[0]):
                    temp = features[i,:]
                    if np.isnan(np.min(temp)):
                        continue
                    else:
                        if feat.size == 0:
                            feat = temp
                        else:
                            feat = np.vstack((feat, temp))
                features = feat;
                features = preprocessing.scale(features)
                return features 
        
            #path to saved models
            modelpath  = "/home/pavan/miniproject/speech_analysis/"     
        
            gmm_files = [os.path.join(modelpath,fname) for fname in 
                        os.listdir(modelpath) if fname.endswith('.gmm')]
            models    = [cPickle.load(open(fname,'rb'),encoding='latin1') for fname in gmm_files]
            genders   = [fname.split("/")[-1].split(".gmm")[0] for fname 
                        in gmm_files]

            sr, audio  = read(pathy)
            features   = get_MFCC(sr,audio)
            scores     = None
            log_likelihood = np.zeros(len(models)) 
            for i in range(len(models)):
                gmm    = models[i]         #checking with each model one by one
                scores = np.array(gmm.score(features))
                log_likelihood[i] = scores.sum()
            winner = np.argmax(log_likelihood)
            if genders[winner]=="male":
                return "Male"
            if genders[winner]=="female":
                return "Female"
        
        

        
        dataPoints=[]
        stt=""
        gcs=request.GET.get("gcs_option");
        if(gcs=="1"):
            gcs="gs://spech-to-text-files/demoaudiofile1.wav"
        elif(gcs=="2"):
            gcs="gs://spech-to-text-files/demoaudiofile2.wav"    
        pathi=request.GET.get("path");
        pathy="/home/pavan/"+  str(pathi)
        if pathi:
            dataPoints,stt,sentences=transcribe_file_with_auto_punctuation(pathy)
            gender=get_gender(pathy)
        elif gcs:
            dataPoints,stt,sentences=transcribe_gcs(gcs)
        value="";
        graph_val=str(dataPoints);
        stt=str(stt);
        
        
        return render_to_response('graph_results.html',locals())
    except Exception as e:
        value=str(e);
        return render_to_response('graph_results.html',locals())

        
	
# Create your views here.

