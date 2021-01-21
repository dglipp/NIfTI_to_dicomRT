import nibabel as nib
import numpy as np
import seaborn as sns
import os
import cv2 as cv
import traceback
import sys
import pydicom
from datetime import date, datetime

def create_color_palette(n):
    '''
    Args:
    ------
        n: Number of RGB colors in palette
    Output:
    -------
        Returns a list of string R\G\B (0, 255) colors
    '''
    palette = sns.color_palette("hls", n)
    str_palette = list()
    for p in palette:
        str_p = '\\'.join([str(int(color*255)) for color in p])
        str_palette.append(str_p)
    return str_palette

def load_Nifti(path):
    '''
    Args:
    ------
        path: path string of the nifti file.
    Output:
    -------
        Returns the 3D matrix of binary segmentation,
        0 is background / 255 is segmented.
        Returns None if any error occurs.
    '''
    try:
        img = nib.load(path)
    except FileNotFoundError:
        print('[ERROR]: File not found.')
        return
    except Exception as err:
        print('[Error]: Unknown error:\n')
        print('\t'+str(err))
        return
    else:
        pixel_data = img.get_fdata()
        if(any(np.sort(np.unique(pixel_data)) != [0., 255.])):
            print('[ERROR]: Binary segmentation values are not 0 or 255.')
            return
        if(len(pixel_data.shape) != 3):
            print('[ERROR]: Pixel map is not 3D.')
            return
        return pixel_data


def load_dicom_filenames(path):
    '''
    Args:
    ------
        path: path string of the dicom CT folder.
    Output:
    -------
        Returns the Dicom filenames as a list of strings.
        Returns None if any error occurs.
    '''
    try:
        folder_els = os.listdir(path)
        dicom_files = [el for el in folder_els if '.dcm' == el[-4:]]

        if (len(dicom_files) > 0):
            return dicom_files
        else:
            print(f'[WARNING]: No Dicom files found in folder "{path}"')
            return []
    except FileNotFoundError:
        print('[ERROR]: File not found.')
        return
    except NotADirectoryError:
        print(f'[ERROR]: "{path}" is not a directory.')
    except Exception as err:
        print('[Error]: Unknown error:\n')
        print('\t'+str(err))
        return

def mask_to_contour(data):
    '''
    Args:
    -------
        data: 3D matrix of a single binary label (0 or 255)
    Output:
    -------
        Assumes the last axis is the number of images.

        Returns an array of lists of contours with shape equals
        to the last dimension of data.
        Returns None if any error occurs.
    '''
    try:
        if (len(data.shape) != 3):
            print(f'[ERROR]: {len(data.shape)}D data not supported.')
            return
        contour_array = np.empty((data.shape[-1],), dtype=object)
        for i in range(data.shape[-1]):
            im_gray = data[:, :, i].astype(np.uint8)
            contours, _ = cv.findContours(
                im_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contours_list = [np.reshape(contour, (contour.shape[0], 2))
                             for contour in contours if contour.shape[0] > 2]
            contour_array[i] = contours_list
        return contour_array
    except AttributeError:
        print('[ERROR]: Not a numpy array.')
        return
    except Exception as err:
        print('[Error]: Unknown error:\n')
        print('\t'+str(err))
        return

def create_affine(dcm_dir):
    '''
    Args:
    -------
    dcm_dir: a dicom dir of the chosen series

    Output:
    -------
        Returns the affine matrix from pixel (voxel) space to 
        patient space
    '''
    affine = np.empty((4,4))

    filenames = np.array(load_dicom_filenames(dcm_dir))
    dcm_z_pos = list()
    for i, dcm_filename in enumerate(filenames):
        dcm_file = pydicom.dcmread(dcm_dir + "/" + dcm_filename)
        dcm_z_pos.append(float(dcm_file.ImagePositionPatient[-1]))

    idx = np.argsort(dcm_z_pos)
    dcm_1 = pydicom.dcmread(dcm_dir + "/" + filenames[idx][0])
    min_z_offset = np.min(dcm_z_pos)
    patient_position = list(np.array(dcm_1.ImagePositionPatient[:2]).astype(float)) + [min_z_offset]
    orientation = np.array(dcm_file.ImageOrientationPatient).astype(int)
    pixel_spacing = np.array(dcm_file.PixelSpacing).astype(float)
    affine[:3,0] = pixel_spacing[1] * orientation[:3]
    affine[:3,1] = pixel_spacing[0] * orientation[3:]

    affine[:3,2] = [0, 0, 1]
    affine[:3,3] = patient_position
    affine[3, :] = [0,0,0,1]
    print(affine)
    return affine

def contour_to_dcm_string(contour, z_value, affine):
    '''
    Args:
    -------
        contour: contour data as np array
        z_value: z value of contour
        affine: affine matrix from pixel to patient space
        
    Output:
    -------
        Returns a string format contour (for RTSTRUCT) 
        Returns None if any error occurs
    '''
    d3_contour = np.zeros((contour.shape[0], contour.shape[1] + 2))
    d3_contour[:,:-2] = contour
    d3_contour[:, [1, 0]] = d3_contour[:, [0, 1]] 
    d3_contour[:, -2] = [z_value] * contour.shape[0]
    d3_contour[:, -1] = [1] * contour.shape[0]
    d3_contour = np.transpose(d3_contour)
    patient_contour = np.matmul(affine, d3_contour)
    patient_contour[2, :] = z_value
    return '\\'.join(patient_contour[:-1, :].flatten('F').astype(str))

def gen_dicom_rt(dicom_dir, nii_paths, order='A'):
    '''
    Args:
    -------
        dicom_dir: path of the dicom CT images folder
        nii_paths: list of nii filepaths
        order: 'A' for ascending order (default)
               'D' for descending order
        
    Output:
    -------
        Returns a pydicom file as a RTSTRUCT
        Returns None if any error occurs
    '''
    try:
        # retrieve dicom filenames from dir
        dicom_filenames = load_dicom_filenames(dicom_dir)

        # read first dicom to obtain common tags
        first_dcm = pydicom.dcmread(dicom_dir + "/" + dicom_filenames[0])

        # create new RT dicom file meta info
        instance_uid = pydicom.uid.generate_uid(prefix='1.2.840.')
        out_dcm = pydicom.dataset.Dataset()
        out_dcm.ensure_file_meta()
        out_dcm.preamble = first_dcm.preamble
        out_dcm.file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
        out_dcm.file_meta.MediaStorageSOPInstanceUID = instance_uid
        out_dcm.file_meta.TransferSyntaxUID = first_dcm.file_meta.TransferSyntaxUID
        pydicom.dataset.validate_file_meta(out_dcm.file_meta, True)
        
        # adding common information

        # SOP Common
        out_dcm.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
        out_dcm.SOPInstanceUID = instance_uid

        # patient
        out_dcm.PatientName = first_dcm.PatientName
        out_dcm.PatientID = first_dcm.PatientID
        out_dcm.PatientBirthDate = first_dcm.PatientBirthDate
        out_dcm.PatientSex = first_dcm.PatientSex
        
        # general study
        out_dcm.StudyDate = first_dcm.StudyDate
        out_dcm.StudyTime = first_dcm.StudyTime
        out_dcm.AccessionNumber = first_dcm.AccessionNumber
        out_dcm.ReferringPhysicianName = first_dcm.ReferringPhysicianName
        out_dcm.StudyInstanceUID = first_dcm.StudyInstanceUID
        out_dcm.StudyID = first_dcm.StudyID

        # RT series
        out_dcm.Modality = 'RTSTRUCT'
        out_dcm.SeriesInstanceUID = pydicom.uid.generate_uid(prefix='1.2.840.')
        out_dcm.SeriesNumber = 1

        # General equipment
        out_dcm.Manufacturer = 'DIGI_LEAP'

        # RT ROI observations
        roi_type_sequence = list()
        for i, nii_filename in enumerate(nii_paths):
            roi_type = pydicom.Dataset()
            roi_type.ObservationNumber = i + 1
            roi_type.ReferencedROINumber = i + 1
            roi_type.RTROIInterpretedType = ''
            roi_type.ROIInterpreter = 'AUTOSEGMENTATION_DNN'
            roi_type.ROIObservationLabel = nii_filename.split('/')[-1].split('.')[-3]
            roi_type_sequence.append(roi_type)

        out_dcm.RTROIObservationsSequence = roi_type_sequence

        # Structure set
        out_dcm.StructureSetLabel = 'AUTOSEGMENTATION_DNN'
        out_dcm.StructureSetDate = date.today().strftime("%Y%m%d")
        out_dcm.StructureSetTime = datetime.now().strftime('%H%M%S')

        structure_set_roi_sequence = list()
        for i, nii_filename in enumerate(nii_paths):
            structure_set_roi = pydicom.Dataset()
            structure_set_roi.ROINumber = i + 1
            structure_set_roi.ReferencedFrameOfReferenceUID = first_dcm.FrameOfReferenceUID
            structure_set_roi.ROIName = nii_filename.split('/')[-1].split('.')[-3]
            structure_set_roi.ROIGenerationAlgorithm = 'AUTOSEGMENTATION_DNN'
            structure_set_roi_sequence.append(structure_set_roi)

        out_dcm.StructureSetROISequence = structure_set_roi_sequence

        contour_image_sequence_ref = list()
        for i, dicom_filename in enumerate(dicom_filenames):
            dicom_file = pydicom.dcmread(dicom_dir + "/" + dicom_filenames[i])

            contour_image_ref = pydicom.Dataset()
            contour_image_ref.ReferencedSOPClassUID = dicom_file.SOPClassUID
            contour_image_ref.ReferencedSOPInstanceUID = dicom_file.SOPInstanceUID
            contour_image_sequence_ref.append(contour_image_ref)
        
        RT_referenced_series = pydicom.Dataset()
        RT_referenced_series.ContourImageSequence = contour_image_sequence_ref
        RT_referenced_series.SeriesInstanceUID = first_dcm.SeriesInstanceUID

        RT_referenced_study = pydicom.Dataset()
        RT_referenced_study.ReferencedSOPClassUID = first_dcm.SOPClassUID
        RT_referenced_study.ReferencedSOPInstanceUID = first_dcm.SOPInstanceUID
        RT_referenced_study.RTReferencedSeriesSequence = [RT_referenced_series]

        referenced_frame_of_reference = pydicom.Dataset()
        referenced_frame_of_reference.FrameOfReferenceUID = first_dcm.FrameOfReferenceUID
        referenced_frame_of_reference.RTReferencedStudySequence = [RT_referenced_study]

        out_dcm.ReferencedFrameOfReferenceSequence = [referenced_frame_of_reference]

        # ROI contour

        dcm_id = np.empty((len(dicom_filenames), ), dtype=object)
        dcm_z_pos = np.empty((len(dicom_filenames), ), dtype=float)
        for i, dcm_filename in enumerate(dicom_filenames):
            dcm_file = pydicom.dcmread(dicom_dir + "/" + dcm_filename)
            dcm_id[i]= [dcm_file.SOPClassUID, dcm_file.SOPInstanceUID]
            dcm_z_pos[i] = float(dcm_file.ImagePositionPatient[-1])

        sort_idx = np.argsort(dcm_z_pos)
        dcm_id = dcm_id[sort_idx]
        dcm_z_pos = dcm_z_pos[sort_idx]

        palette = create_color_palette(len(nii_paths))

        affine_matrix = create_affine(dicom_dir)
        
        roi_contour_sequence = list()
        for i, nii_filename in enumerate(nii_paths):
            roi_contour = pydicom.Dataset()
            roi_contour.ReferencedROINumber = i + 1
            roi_contour.ROIDisplayColor = palette[i]

            seg_data = mask_to_contour(nib.load(nii_filename).get_fdata())

            contour_sequence = list()
            for i, dicom_filename in enumerate(dicom_filenames):
                for closed_contour in seg_data[i]:
                    contour = pydicom.Dataset()

                    contour.ContourGeometricType = 'CLOSED_PLANAR'
                    contour.NumberOfContourPoints = len(closed_contour)

                    contour.ContourData = contour_to_dcm_string(closed_contour, dcm_z_pos[i], affine_matrix)

                    image_sequence = pydicom.Dataset()
                    image_sequence.ReferencedSOPClassUID = dcm_id[i][0]
                    image_sequence.ReferencedSOPInstanceUID = dcm_id[i][1]
                    contour.ContourImageSequence = [image_sequence]
                    
                    contour_sequence.append(contour)

            roi_contour.ContourSequence = contour_sequence
            roi_contour_sequence.append(roi_contour)
        out_dcm.ROIContourSequence = roi_contour_sequence
        return out_dcm

    except FileNotFoundError:
        print('[ERROR]: File not found.')
        return None
    except ValueError as err:
        print('[ERROR]: Missing information in Dicom input files:')
        print('\t'+str(err))
    except Exception as err:
        print('[ERROR]: Unknown error:\n')
        print('\t'+str(err))
        return None



nii_folder = 'data/patient_temp/results'

nii_path = [nii_folder + "/" + nii_filename for nii_filename in os.listdir(nii_folder)]

dicom_path = 'data/patient_temp/patientTac/1.2.840.113704.7.1.0.16716261142165132.1606405308.0/1.2.840.113704.7.1.0.16716261142165132.1606405309.4'

gen_dicom_rt(dicom_path, nii_path).save_as('test_to_remove.dcm')


