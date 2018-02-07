import random
import sys
import os


def load_stage_anno(stage_anno_file):
    stage_anno = {}
    with open(stage_anno_file) as fin:
        for i, line in enumerate(fin):
            if i == 0:
                continue
            patient, stage = line.strip().split(',')
            patient_id = patient[:-4].split('_')[1]
            stage_anno.setdefault(patient_id, []).append((patient, stage))
    return stage_anno

def get_split_list():
    stage_anno_file = '../dataset/train/stage_labels.csv'
    stage_anno = load_stage_anno(stage_anno_file)

    xml_root = '../dataset/train/label/'
    xml_files = sorted(os.listdir(xml_root))
    anno_patient_id = [xml_file.split('_')[1] for xml_file in xml_files]
    anno_patient_id = sorted(list(set(anno_patient_id)))

    train_xml = [ xml_files[i][:-4] for i in range(len(xml_files)) if i % 10 != 0 ]
    test_xml = [ xml_files[i][:-4] for i in range(len(xml_files)) if i % 10 == 0 ]

    with open('train_set.txt', 'w') as fout:
        for patient_id in anno_patient_id:
            records = stage_anno[patient_id]
            for patient, stage in records:
                if patient[-3:] == 'zip':
                    continue
                patient = patient[:-4]
                rtype = 'None'
                if patient in train_xml:
                    rtype = 'train'
                if patient in test_xml:
                    rtype = 'test'
                if rtype == 'None' and stage != 'negative':
                    continue
                fout.write('%s\t%s\t%s\n' % (patient, stage, rtype))

    with open('val_set.txt', 'w') as fout:
        for patient_id, records in stage_anno.items():
            if patient_id in anno_patient_id:
                continue
            for patient, stage in records:
                fout.write('%s\t%s\n' % ( patient, stage ))


if __name__ == '__main__':
    get_split_list()
