import parsl
from parsl.app.app import python_app
from parsl.app.app import bash_app
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.providers import AdHocProvider
from parsl.channels import LocalChannel
from parsl.data_provider.files import File
from parsl.channels import SSHChannel
from parsl.app.errors import BashExitFailure
import networkx as nx
import itertools
import os.path



local_thing = Config(
    executors=[
        HighThroughputExecutor(
            label="local_htex",
            worker_debug=False,
            max_workers=48,
            provider=LocalProvider(
                channel=LocalChannel(),
                init_blocks=1,
                max_blocks=1,
            ),
        )
    ]
)
parsl.load(local_thing)

@bash_app 
def sed(inputs = [], outputs=[]):
    #sed -i \'s/>>graph6<<//g\' 14-graphs/14iso0.g6
    #return "echo " + inputs[0].filepath + " > " +  outputs[0].filepath
    return "sed -i \'s/>>graph6<<//g\' " + inputs[0].filepath

@bash_app
def nauty(inputs = [], outputs=[]):
    #../../../nauty2_8_6/shortg -g 14-graphs/14iso0.g6 14-isos/14out0.g6
    return "../../../nauty2_8_6/shortg -g " + inputs[0].filepath + " " +  outputs[0].filepath



####
####PART 1
# jobs = {}
# outputs = []
# for i in range(0, 133):
#     file_name = "14-graphs/14iso" + str(i) + ".g6" 
#     #file_name = "test_thing" + str(i) + ".txt"
#     if os.path.isfile(file_name):
#         print(file_name)
#         jobs[i] = sed(inputs=[File(os.path.join(os.getcwd(), file_name))], outputs=[File(os.path.join(os.getcwd(), file_name))])
#         job = jobs[i]
#         try:
#             outputs.append( (i, job.result()) )
#         except BashExitFailure as inst:
#             print("Failure" + str(i))
#             print(inst, str(job))
# print(outputs)

####
####PART 2
# jobs = {}
# outputs = []
# for i in range(0, 133):
#     input_name = "14-graphs/14iso" + str(i) + ".g6" 
#     output_name = "14-isos/14out" + str(i) + ".g6" 
#     #file_name = "test_thing" + str(i) + ".txt"
#     if os.path.isfile(input_name):
#         print(input_name)
#         jobs[i] = nauty(inputs=[File(os.path.join(os.getcwd(), input_name))], outputs=[File(os.path.join(os.getcwd(), output_name))] )
#         #print(jobs[i])
#         job = jobs[i]
#         try:
#             outputs.append( (i, job.result()) )
#         except BashExitFailure as inst:
#             print("Failure" + str(i))
#             print(inst, str(job))
# print(outputs)


##EXTRA STUFF FOR SPLITTING CASES

# input_names = [
#     "splits/14iso115partaa.g6", "splits/14iso115partab.g6", "splits/14iso115partac.g6", "splits/14iso115partad.g6",
#     "splits/14iso115partae.g6", "splits/14iso115partaf.g6",
#     "splits/14iso116partaa.g6", "splits/14iso116partab.g6", "splits/14iso116partac.g6", "splits/14iso116partad.g6",
#     "splits/14iso116partae.g6", "splits/14iso116partaf.g6",
#     "splits/14iso130partaa.g6", "splits/14iso130partab.g6", "splits/14iso130partac.g6", "splits/14iso130partad.g6",
#     "splits/14iso130partae.g6", "splits/14iso130partaf.g6", 
#     "splits/14iso131partaa.g6", "splits/14iso131partab.g6", "splits/14iso131partac.g6", "splits/14iso131partad.g6",
#     "splits/14iso131partae.g6", "splits/14iso131partaf.g6"
# ]
# output_names = [
#     "splits-out/14out115part1.g6", "splits-out/14out115part2.g6", "splits-out/14out115part3.g6", "splits-out/14out115part4.g6",
#     "splits-out/14out115part5.g6", "splits-out/14out115part6.g6",
#     "splits-out/14out116part1.g6", "splits-out/14out116part2.g6", "splits-out/14out116part3.g6", "splits-out/14out116part4.g6",
#     "splits-out/14out116part5.g6", "splits-out/14out116part6.g6",
#     "splits-out/14out130part1.g6", "splits-out/14out130part2.g6", "splits-out/14out130part3.g6", "splits-out/14out130part4.g6",
#     "splits-out/14out130part5.g6", "splits-out/14out130part6.g6",
#     "splits-out/14out131part1.g6", "splits-out/14out131part2.g6", "splits-out/14out131part3.g6", "splits-out/14out131part4.g6",
#     "splits-out/14out131part5.g6", "splits-out/14out131part6.g6"
# ]
# #print(inputs)
# jobs = []
# outputs = []
# for i in range(len(input_names)):
#     print(input_names[i])
#     jobs.append( nauty(inputs=[File(os.path.join(os.getcwd(), input_names[i]))], outputs=[File(os.path.join(os.getcwd(), output_names[i])) ] ) )
#     #print(jobs[i])
#     job = jobs[i]
#     try:
#         outputs.append( (i, job.result()) )
#     except BashExitFailure as inst:
#         print("Failure" + str(i))
#         print(inst, str(job))

#COMBINING STUFF
import shutil
import subprocess
import os

base = 0
working_file_name = ""
isos_file_name = ""
for curr in range(0, 133):
    print("curr:", curr)
    curr_file_name = "14-isos/14out" + str(curr) + ".g6" 
    if curr == 0:
        working_file_name = "combined-graphs/comb" + str(base)+ ".g6"
        isos_file_name = "combined-graphs/finished-isos/isos" + str(base) + "-" + str(curr) + ".g6"
        shutil.copyfile(curr_file_name, working_file_name)
        shutil.copyfile(curr_file_name, isos_file_name)
    else:
        #Truncates the existing file
        new_working_file = open(working_file_name, "w")
        new_iso_file_name = "combined-graphs/finished-isos/isos" + str(base) + "-" + str(curr) + ".g6"
        subprocess.run(["../../../nauty2_8_6/catg", isos_file_name, curr_file_name], stdout=new_working_file)
        try:
            subprocess.run(["../../../nauty2_8_6/shortg", "-g", working_file_name, new_iso_file_name], check=True)
            os.remove(isos_file_name)
            isos_file_name = new_iso_file_name
            print("Made isos", base, " to ", curr)
        except:
            #this means the short g failed, new part to make
            #old isos_file_name can exist and stay
            base = curr
            working_file_name = "combined-graphs/comb" + str(base)+ ".g6"
            isos_file_name = "combined-graphs/finished-isos/isos" + str(base) + "-" + str(curr) + ".g6"
            shutil.copyfile(curr_file_name, working_file_name)
            shutil.copyfile(curr_file_name, isos_file_name)
            print("New base", curr)