import os
os.chdir('/Users/giosue/projects/DicomConverter')
seglist = os.listdir('data/patient_temp/results')
seglist = [('data/patient_temp/results/'+el) for el in seglist]
# seglist = ','.join(seglist)
os.system('dcmqi/bin/itkimage2segimage --inputImageList {} --inputDICOMDirectory data/patient_temp/patientTac/1.2.840.113704.7.1.0.16716261142165132.1606405308.0/ --outputDICOM outputDicomTest.dcm --inputMetadata data/patient_temp/report.json'.format(seglist[0]))
