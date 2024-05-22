from Bio import pairwise2
from Bio.Align import substitution_matrices
import numpy as np

cluster =  {6 : ['FLYALALLL', 'LLLDDFVEI', 'TTDPSFLGRY', 'PTDNYITTY', 'IPTITQMNL', 'VLSFCAFAV', 'RLPGVLPRA', 'RAKFKQLL', 'IVTDFSVIK', 'MEVTPSGTWL', 'RYSIFFDYM', 'PQPELPYPQPQL', 'NTNSSPDDQIGYY', 'LLFGYPVYV', 'CTELKLSDY', 'YVLDHLIVV', 'NLVPMVATV', 'GILGFVFTL', 'VAANIVLTV', 'LLWNGPMAV', 'SVLYYQNNV', 'VYFLQSINF', 'TLKNTVCTV', 'YLLEMLWRL', 'GPRLGVRAT', 'KPYIKWDLL', 'KLQFTSLEI', 'VYAWNRKRI', 'KLVALVINAV', 'LMNVLTLVY', 'WLTNIFGTV', 'KLVALGINAV', 'KLGGALQAK', 'AYAQKIFKI', 'LTDEMIAQY', 'GLCTLVAML', 'YLEPGPVTA', 'RQLLFVVEV', 'RLRAEAQVK', 'FLNGSCGSV', 'ELAGIGILTV', 'QYIKWPWYI', 'IITTDNTFV', 'SPRWYFYYL', 'FSNVTWFHA', 'KLVAMGINAV', 'ILFTRFFYV', 'AVFDRKSDAK', 'TFEYVSQPFLMDLE', 'DATYQRTRALVR', 'ALDPHSGHFV', 'CINGVCWTV', 'YHSIEWA', 'YLQPRTFLL', 'YQDVNCTEV'],
            5 : ['RLARLALVL', 'GTDLEGNFY', 'TTDPSFLGRY', 'PTDNYITTY', 'SSGDATTAY', 'FTVLCLTPV', 'RLPGVLPRA', 'YVVDDPCPI', 'RAKFKQLL', 'IVTDFSVIK', 'HYNYMCNSSCMGSMN', 'YVDDSSLTI', 'PQPELPYPQPE', 'PGVLLKEFTVSGNIL', 'NLVPMVATV', 'GILGFVFTL', 'RLITGRLQSL', 'PFPQPELPY', 'EAAGIGILTV', 'APRGPHGGAASGL', 'KPASRELKV', 'LLWNGPMAV', 'KTFPPTEPK', 'KQIYKTPPI', 'SMMILSDDA', 'NVLTLVYKV', 'WMESEFRVY', 'YLLEMLWRL', 'IPVAYRKVL', 'IMDQVPFSV', 'YVFCTVNAL', 'KTWGQYWQV', 'RIMTWLDMV', 'LLLDRLNQL', 'KLSALGINAV', 'KLVALGINAV', 'SLTYSTAAL', 'FLPRVFSAV', 'FITESKPSV', 'KLGGALQAK', 'KVSIWNLDY', 'TPRVTGGGAM', 'TWLTYTGAI', 'AYAQKIFKI', 'GLCTLVAML', 'LTDEMIAQY', 'DTDFVNEFY', 'VPYFNMVYM', 'KSVNITFEL', 'RLRAEAQVK', 'NQKLIANQF', 'ELAGIGILTV', 'IPIQASLPF', 'TDDNALSYY', 'LSDRVVFVL', 'SPRWYFYYL', 'SANNCTFEY', 'KLVAMGINAV', 'AVFDRKSDAK', 'DATYQRTRALVR', 'QVILLNKHI', 'CINGVCWTV', 'LMCQPILLL', 'ALWEIQQVV', 'YLQPRTFLL', 'VLSTFISAA', 'LPRWYFYYL'],
            4 : ['MLDLQPETT', 'RLGEVRHPV', 'RTAPHGHVV', 'TTDPSFLGRY', 'RLPGVLPRA', 'RAKFKQLL', 'IVTDFSVIK', 'YYRYNLPTM', 'MEVTPSGTWL', 'RPPIFIRRL', 'KVLEYVIKV', 'ALSGVFCGV', 'PQPELPYPQPQL', 'QADVEWKFY', 'PGVLLKEFTVSGNIL', 'NLVPMVATV', 'GILGFVFTL', 'KVDPIGHVY', 'YGFQPTNGV', 'EAAGIGILTV', 'LLMPILTLT', 'SLFNTVATLY', 'SLFNTVATL', 'LLWNGPMAV', 'ILRKGGRTI', 'KTFPPTEPK', 'APRGPHGGAASGL', 'VYFLQSINF', 'ALLPGLPAA', 'SMMILSDDA', 'YHGAIKLDD', 'GLALYYPSA', 'MTLHGHMMY', 'VLGSLAATV', 'PKYVKQNTLKLAT', 'FLPGVYSVI', 'KVYPIILRL', 'KLVALGINAV', 'VFVQSVLPYFVATKLAKIRK', 'KLGGALQAK', 'VPYFNMVYM', 'GELIGILNAAKVPAD', 'AYAQKIFKI', 'GLCTLVAML', 'LTDEMIAQY', 'KLYGLDWAEL', 'KSVNITFEL', 'RLRAEAQVK', 'ELAGIGILTV', 'NLNCCSVPV', 'QYIKWPWYI', 'LWLLWPVTL', 'QMAPISAMV', 'SPRWYFYYL', 'AVFDRKSDAK', 'DATYQRTRALVR', 'IPSINVHHY', 'MMILSDDAV', 'CINGVCWTV', 'ALWEIQQVV', 'YHSIEWA', 'YLQPRTFLL', 'FTPLVPFWI', 'GLPWNVVRI', 'LLDFVRFMGV'],
            0 : ['TTDPSFLGRY', 'RAKFKQLL', 'IVTDFSVIK', 'RTLNAWVKV', 'YADVFHLYL', 'MEVTPSGTWL', 'NTNSSPDDQIGYY', 'PGVLLKEFTVSGNIL', 'LLEWLAMAV', 'NLVPMVATV', 'GILGFVFTL', 'YVLDHLIVV', 'SLFNTVATLY', 'HPVTKYIM', 'LLWNGPMAV', 'KTFPPTEPK', 'YPDKVFRSSV', 'ALLPGLPAA', 'KLFEFLVYGV', 'PKYVKQNTLKLAT', 'KLVALGINAV', 'LYALVYFLQ', 'KLGGALQAK', 'QELIRQGTDY', 'AYAQKIFKI', 'LTDEMIAQY', 'GLCTLVAML', 'ALSKGVHFV', 'RLRAEAQVK', 'ELAGIGILTV', 'QYIKWPWYI', 'APATVCGPK', 'SPRWYFYYL', 'KAFSPEVIPMF', 'KLVAMGINAV', 'AVFDRKSDAK', 'DATYQRTRALVR', 'CINGVCWTV', 'YLQPRTFLL', 'FTPLVPFWI'],
            1 : ['SLLMWITQV', 'MMISAGFSL', 'FLYALALLL', 'TTDPSFLGRY', 'RLPGVLPRA', 'RAKFKQLL', 'IVTDFSVIK', 'NLVPMVATV', 'GILGFVFTL', 'HPVTKYIM', 'LLWNGPMAV', 'PKYVKQNTLKLAT', 'KLVALGINAV', 'KLGGALQAK', 'GLCTLVAML', 'RLRAEAQVK', 'NQKLIANQF', 'ELAGIGILTV', 'SPRWYFYYL', 'AVFDRKSDAK', 'CINGVCWTV'],
            3 : ['TTDPSFLGRY', 'VVMSWAPPV', 'RAKFKQLL', 'IVTDFSVIK', 'PQPELPYPQPQL', 'FISTCACEI', 'FLRGRAYGL', 'NLVPMVATV', 'GILGFVFTL', 'VAANIVLTV', 'LLWNGPMAV', 'SVLYYQNNV', 'KTFPPTEPK', 'LDDKDPNFK', 'VYFLQSINF', 'FLALCADSI', 'GPRLGVRAT', 'KLFEFLVYGV', 'LLFNKVTLA', 'KTWGQYWQV', 'KLWAQCVQL', 'KLVALGINAV', 'KLGGALQAK', 'AYAQKIFKI', 'LTDEMIAQY', 'GLCTLVAML', 'KSVNITFEL', 'ELAGIGILTV', 'QYIKWPWYI', 'AEVQIDRLI', 'SPRWYFYYL', 'KLVAMGINAV', 'ALDPHSGHFV', 'AVFDRKSDAK', 'LSFKELLVY', 'ALWEIQQVV', 'YLQPRTFLL', 'KDNVILLNK'],
            2 : ['TTDPSFLGRY', 'RLPGVLPRA', 'RAKFKQLL', 'IVTDFSVIK', 'HPVGEADYFEY', 'NTNSSPDDQIGYY', 'RLGPVQNEV', 'NLVPMVATV', 'GILGFVFTL', 'LLWNGPMAV', 'KTFPPTEPK', 'QPGQTFSVL', 'LLLDRLNQL', 'MVTNNTFTL', 'KLGGALQAK', 'FLIGCNYLG', 'FVCDNIKFA', 'NYNYLYRLF', 'KSKRTPMGF', 'RSRNSSRNL', 'ELAGIGILTV', 'NYMPYFFTL', 'SPRWYFYYL', 'TFEYVSQPFLMDLE', 'AVFDRKSDAK', 'CINGVCWTV', 'YLQPRTFLL']}


def calculate_similarity(epitope1, epitope2):
    # Load the BLOSUM62 matrix
    matrix = substitution_matrices.load("BLOSUM62")
    
    # Perform a global alignment using the BLOSUM62 matrix
    alignments = pairwise2.align.globaldx(epitope1, epitope2, matrix)
    
    # Select the best alignment (highest score)
    top_alignment = alignments[0]
    aligned_epitope1, aligned_epitope2, score, begin, end = top_alignment
    
    return score

closest_pairs = {}

for key, epitopes in cluster.items():
    max_score = float('-inf')
    closest_pair = None
    
    n = len(epitopes)
    # Calculate pairwise similarity for each pair of epitopes
    for i in range(n):
        for j in range(i + 1, n):  # Ensure each pair is only calculated once
            score = calculate_similarity(epitopes[i], epitopes[j])
            # Check if the current score is the highest found so far
            if score > max_score:
                max_score = score
                closest_pair = (epitopes[i], epitopes[j])
    
    # Store the result
    if closest_pair:
        closest_pairs[key] = (closest_pair, max_score)
    else:
        closest_pairs[key] = ("No sufficient pairs", 0)

# Output the results
for key, value in closest_pairs.items():
    epitope_pair, score = value
    print(f"Closest pair in cluster {key} with similarity score {score}: {epitope_pair}")

# for key, epitopes in cluster.items():
#     min_score = float('inf')  # Initialize min_score to positive infinity
#     closest_pair = None
    
#     n = len(epitopes)
#     # Calculate pairwise similarity for each pair of epitopes
#     for i in range(n):
#         for j in range(i + 1, n):  # Ensure each pair is only calculated once
#             score = calculate_similarity(epitopes[i], epitopes[j])
#             # Check if the current score is the lowest found so far
#             if score < min_score:
#                 min_score = score
#                 closest_pair = (epitopes[i], epitopes[j])
    
#     # Store the result
#     if closest_pair:
#         closest_pairs[key] = (closest_pair, min_score)
#     else:
#         closest_pairs[key] = ("No sufficient pairs", 0)

# # Output the results
# for key, value in closest_pairs.items():
#     epitope_pair, score = value
#     print(f"Closest pair in cluster {key} with similarity score {score}: {epitope_pair}")