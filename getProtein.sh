PDB=$1
curl -O https://files.rcsb.org/view/${PDB}.pdb
grep "ATOM " ${PDB}.pdb | awk '{print $3, $4, $5, $7, $8, $9}' > atoms.txt

OIFS=$IFS
IFS='\n'
while read line; do
    res=$(echo $line | cut -d ' ' -f 2)
    atom=$(echo $line | cut -d ' ' -f 1)
    start=$(grep -n "\[ ${res} ]" aminoacids.txt | cut -d: -f1)
    tail -n +${start} aminoacids.txt | grep "${atom}" | head -1 | awk '{print $3}' >> charge.txt    
done < atoms.txt
IFS=$OIFS

awk '{print $4, $5, $6}' atoms.txt > positions.txt

rm ${PDB}.pdb atoms.txt
