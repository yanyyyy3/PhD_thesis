#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:16:59 2019

@author: yanying
"""

import numpy as np
import argparse
from Bio import SeqIO
from Bio.Seq import Seq
import subprocess
import os
import time 
import datetime
import logging
import itertools
import pandas
import sys
from tqdm import tqdm

start_time=time.time()
st = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)
parser = MyParser(usage='python %(prog)s gRNA CSV file [options]',formatter_class=argparse.RawTextHelpFormatter,description="""
This is used to compute extensive features for gRNAs. Including: 4 thermodynamic features (MFE features), gene information 

For the input file, CSV format is accepted. Columns with name 'target' for gRNA sequence are required.  

2 versions of Vienna packages are required. For installation, please check the information in the repository.

Example: python feature_engineering.py test.csv FASTA GFF3 -o test
                  """)
parser.add_argument("library", help="gRNA library csv file")
parser.add_argument("FASTA",  help="FASTA file for reference/assembly genome ")
parser.add_argument("GFF3",  help="GFF3 file for reference/assembly genome ")
parser.add_argument("-o", "--output", default="results", help="output file name")
parser.add_argument("-g", "--genome_format", default="reference", help="genome format: reference or contigs ")
parser.add_argument("-b", "--before", type=int,default=20, help="the number of base pairs upstream of gRNAs to be extended")
parser.add_argument("-a", "--after",type=int, default=20, help="the number of base pairs downstream of gRNAs to be extended")

args = parser.parse_args()
library_df=args.library
FASTA=args.FASTA
GFF3=args.GFF3
output_file_name = args.output
genome_format=args.genome_format
before=args.before
after=args.after

try:
    os.mkdir(output_file_name)
except:
    overwrite=input("File exists, do you want to overwrite? (y/n)")
    if overwrite == "y":
        os.system("rm -r %s"%output_file_name)
        os.mkdir(output_file_name)
    elif overwrite =="n":
        output_file_name=input("Please give a new output file name:")
        os.mkdir(output_file_name)
    else:
        print("Please input valid choice..\nAbort.")
        sys.exit()
repeat='gtcaaaagacctttttaatttctactcttgtagat'
repeat=repeat.upper()

def consecutive_nt_calculation(sequence):
    maxlen=0
    for k,g in itertools.groupby(sequence):
        group=list(g)
        if len(group)>maxlen:
            maxlen=len(group)
    return maxlen

def MFE_RNA_RNA_hybridization(sequence1,sequence2):
    with open(output_file_name+"/MFE_hybridization.fasta","w") as MFE_hybridization_fasta:
        MFE_hybridization_fasta.write(">s1\n"+sequence1+"\n>s2\n"+sequence2+"\n")
    hybridization_file=open(output_file_name + '/hybridization.txt',"w")
    hybridization_fasta=open(output_file_name+"/MFE_hybridization.fasta",'r')
    subprocess.run(["RNAduplex2.4.14"],stdin=hybridization_fasta,stdout=hybridization_file)
    for line in open(output_file_name + '/hybridization.txt'):
        if ":" in line:
            MFE=line.split(":")[1].split("(")[1].split(")")[0]  
    return MFE

def MFE_RNA_DNA_hybridization(sequence1,sequence2):
    with open(output_file_name+"/MFE_hybridization_DNA.fasta","w") as MFE_hybridization_fasta:
        MFE_hybridization_fasta.write(">s1_RNA\n"+sequence1+"\n>s2_DNA\n"+sequence2+"\n")  # first RNA and then DNA
    hybridization_file=open(output_file_name + '/hybridization.txt',"w")
    hybridization_fasta=open(output_file_name+"/MFE_hybridization_DNA.fasta",'r')
    subprocess.run(["RNAduplex2.1.9h"],stdin=hybridization_fasta,stdout=hybridization_file)
    for line in open(output_file_name + '/hybridization.txt'):
        if ":" in line:
            MFE=line.split(":")[1].split("(")[1].split(")")[0]  
    return MFE

def MFE_folding(sequence):
    with open(output_file_name+"/MFE_folding.fasta","w") as MFE_folding_fasta:
        MFE_folding_fasta.write(">s\n"+sequence+"\n")
    folding_file=open(output_file_name + '/folding.txt',"w")
    subprocess.run(["RNAfold2.4.14","--noPS","-i",output_file_name+"/MFE_folding.fasta"],stdout=folding_file)
    for line in open(output_file_name + '/folding.txt'):
        if "-" in line or "+" in line or "0.00" in line:
            MFE=line.split("(")[-1].split(")")[0]
    subprocess.run(["rm",output_file_name + '/folding.txt',output_file_name+"/MFE_folding.fasta"])
    return MFE


def main():
    open(output_file_name + '/log.txt','a').write("Python script: %s\n"%sys.argv[0])
    open(output_file_name + '/log.txt','a').write("Parsed arguments: %s\n\n"%args)
    ## import FASTA file
    fasta_sequences = SeqIO.parse(open(FASTA),'fasta')    
    
    if genome_format=='reference':
        for fasta in fasta_sequences:  # input reference genome
            reference_fasta=fasta.seq 
    elif genome_format=='contigs':
        reference_fasta=dict()
        for fasta in fasta_sequences:  # input reference genome
            reference_fasta.update({fasta.id:fasta.seq}) 
    GFF=dict()
    for line in open(GFF3):
        if "#" not in line and ("CDS" in line or "pseudogenic_exon" in line or 'RNA' in line):
            line=line.replace("\n","")
            row=line.split("\t")
            if row[2]!='CDS' and row[2]!="pseudogenic_exon" and row[2] !='tRNA' and row[2] !='rRNA':
                continue
            genome=str(row[0])
            geneid=row[8].split("ID=")[1].split(".")[0]
            genename=row[8].split("Name=")[1].split(";")[0]
            length=int(row[4])-int(row[3])+1
            genetype=row[2]
            product=row[8].split("product=")[1].split(";")[0]
            if "protein_id" in row[8]:
                protein_id=row[8].split("protein_id=")[1].split(";")[0]
            else:
                protein_id='NaN'
            if genome_format=='reference':
                seq=reference_fasta[int(row[3])-1:int(row[4])]
            else:
                seq=reference_fasta[genome][int(row[3])-1:int(row[4])]
            if row[6]=="+":
                strand=1
                seq_5_3=seq    
                start=int(row[3])
                end=int(row[4])      
            elif row[6]=="-":
                strand=-1
                seq_5_3=seq.reverse_complement()   
                start=int(row[4])
                end=int(row[3])    
            GC_content = round((seq.count('G') + seq.count('C')) / len(seq) * 100,2)
            if genome_format=='reference':
                GFF.update({tuple([int(row[3]),int(row[4])]):{'genome':genome,"genename":genename,"geneid":geneid,
                                                              'protein_id':protein_id,'genetype':genetype,
                                                        "start":start,"end":end,
                                "GC_content":GC_content,"strand":strand,"length":length,"product":product,'seq_5_3':seq_5_3}})
            else:
                if genome not in GFF.keys():
                    GFF[genome]=dict()
                else:
                    GFF[genome].update({tuple([int(row[3]),int(row[4])]):{'genome':genome,"genename":genename,
                                                                      "geneid":geneid,'protein_id':protein_id,'genetype':genetype,
                                                        "start":start,"end":end,"GC_content":GC_content,
                                                        "strand":strand,"length":length,"product":product,
                                                        'seq_5_3':seq_5_3}})

        elif "##FASTA" in line:
            break
    
    ### import library 
    library=pandas.read_csv(library_df,sep="\t",dtype={'strand':str,'pos':int})
    library=library[library['id']!='Not found']
    open(output_file_name + '/log.txt','a').write("Number of targeting guides: %s\n"%library.shape[0])
    intergenic_count=0
    for i in tqdm(library.index):
        sequence=library['target'][i]
        genome=library['id'][i]
        genome_pos=library['pos'][i]
        guide_strand=library['strand'][i]
        
        if genome_format=='reference':
            if guide_strand=='+':
                guide_strand=1
                PAM=reference_fasta[genome_pos-4:genome_pos]
                extended_seq=reference_fasta[genome_pos-before:genome_pos+len(sequence)+after]
                if sequence != reference_fasta[genome_pos:genome_pos+len(sequence)]:
                    print(sequence,reference_fasta[genome_pos:genome_pos+len(sequence)])
                    sys.exit()
            elif guide_strand=='-':
                guide_strand=-1
                PAM=reference_fasta[genome_pos:genome_pos+4].reverse_complement()
                extended_seq=str(reference_fasta[genome_pos-len(sequence)-after:genome_pos+before].reverse_complement())
                if sequence != str(reference_fasta[genome_pos-len(sequence):genome_pos].reverse_complement()):
                    print(sequence,reference_fasta[genome_pos-len(sequence):genome_pos])
                    sys.exit()
        elif genome_format=='contigs':
            if guide_strand=='+':
                guide_strand=1
                PAM=reference_fasta[genome][genome_pos-4:genome_pos]
                extended_seq=reference_fasta[genome][genome_pos-before:genome_pos+len(sequence)+after]
                if sequence != reference_fasta[genome][genome_pos:genome_pos+len(sequence)]:
                    print(sequence,reference_fasta[genome][genome_pos:genome_pos+len(sequence)])
                    sys.exit()
            elif guide_strand=='-':
                guide_strand=-1
                PAM=reference_fasta[genome][genome_pos:genome_pos+4].reverse_complement()
                extended_seq=str(reference_fasta[genome][genome_pos-len(sequence)-after:genome_pos+before].reverse_complement())
                if sequence != str(reference_fasta[genome][genome_pos-len(sequence):genome_pos].reverse_complement()):
                    print(sequence,reference_fasta[genome][genome_pos-len(sequence):genome_pos].reverse_complement())
                    sys.exit()
        # if PAM[:3] !="TTT":
        #     print(PAM)
        #     sys.exit()         
        genome_pos_5_end=genome_pos
        genome_pos_3_end=genome_pos_5_end+guide_strand*len(sequence)
        library.at[i,'PAM']=PAM
        gene=0
        if genome_format=='reference':
            for loc in GFF.keys():
                if (genome_pos_5_end <= loc[1] and genome_pos_5_end >= loc[0]) or (genome_pos_3_end <= loc[1] and genome_pos_3_end >= loc[0]):
                    gene=GFF[loc]
                    break
            
        else:
            if genome in GFF.keys():
                for loc in GFF[genome].keys():
                    if (genome_pos_5_end <= loc[1] and genome_pos_5_end >= loc[0]) or (genome_pos_3_end <= loc[1] and genome_pos_3_end >= loc[0]):
                        gene=GFF[genome][loc]
                        break
            else:
                print(genome,"not in GFF keys")
        if type(gene)==int:
            intergenic_count+=1
            library.at[i,'genetype']="intergenic"
            # print(sequence, "targeting intergenic location.")
        else:
            gene_strand=gene["strand"]
            gene_5=gene['start']
            library.at[i,'genename']=gene["genename"]
            library.at[i,'geneid']=gene["geneid"]
            library.at[i,'protein_id']=gene["protein_id"]
            library.at[i,'genetype']=gene["genetype"]
            library.at[i,'start']=gene_5
            library.at[i,'end']=gene['end']
            library.at[i,'gene_length']=gene['length']
            library.at[i,'gene_strand']=gene["strand"]
            library.at[i,'gene_GC_content']=gene['GC_content']
            library.at[i,'product']=gene['product']
            
            if guide_strand==gene_strand:
                library.at[i,'coding_strand']=0
            else:
                library.at[i,'coding_strand']=1
            
            distance_start_codon=min(int(gene_strand)*(genome_pos_3_end-gene_5),int(gene_strand)*(genome_pos_5_end-gene_5))
            distance_start_codon_perc=distance_start_codon/gene['length']*100
            library.at[i,'distance_start_codon']=distance_start_codon
            library.at[i,'distance_start_codon_perc']=distance_start_codon_perc
            
        target_seq=str(Seq(sequence).reverse_complement())
        library.at[i,'guide_GC_content']='{:.2f}'.format((sequence.count('G') + sequence.count('C')) / len(sequence) * 100)
        #"MFE_hybrid_full","MFE_hybrid_seed","MFE_mono_guide","MFE_dimer_guide"
        library.at[i,'MFE_hybrid_full']=MFE_RNA_DNA_hybridization(sequence.replace("T","U"),target_seq)
        # library.at[i,'MFE_hybrid_seed']=MFE_RNA_DNA_hybridization(sequence[-8:].replace("T","U"),target_seq[:8])
        library.at[i,'MFE_homodimer_guide']=MFE_RNA_RNA_hybridization(sequence,sequence)
        library.at[i,'MFE_monomer_guide']=MFE_folding(sequence)
        # with repeat
        library.at[i,'MFE_homodimer_guide_repeat']=MFE_RNA_RNA_hybridization(repeat+sequence,repeat+sequence)
        library.at[i,'MFE_monomer_guide_repeat']=MFE_folding(repeat+sequence)
        
        #"consective_nts"
        library.at[i,'homopolymers']=consecutive_nt_calculation(sequence)
        library.at[i,'extended_seq']=extended_seq
    
    library.to_csv(output_file_name+"/gRNAs.csv",sep='\t',index=False)  
    logging.info("The number of targeting guides targeting intergenic region: {0}".format(intergenic_count))
    subprocess.run(["rm",output_file_name+"/MFE_hybridization.fasta",output_file_name + '/hybridization.txt',output_file_name+"/MFE_hybridization_DNA.fasta"]) 
if __name__ == '__main__':
    logging_file= output_file_name + '/log.txt'
    logging.basicConfig(filename=logging_file,format='%(asctime)s - %(message)s', level=logging.INFO)
    main()
    logging.info("Execution Time: %s seconds\n" %('{:.2f}'.format(time.time()-start_time)))    
