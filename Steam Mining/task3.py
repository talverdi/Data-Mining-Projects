from blackbox import BlackBox
import random
import sys
import time

start=time.time()

s=100
glob_list=[]
final=[]
def main(input_file,stream_size,num_of_asks,output_file):
    global glob_list,s
    random.seed(553)
    bx=BlackBox()
    #stream_users=bx.ask('users.txt', stream_size)
    #print("seqnum,0_id,20_id,40_id,60_id,80_id")
    #print(s,stream_users[0],stream_users[20],stream_users[40],stream_users[60],stream_users[80])
    user_count=0
    for ask in range(1,num_of_asks+1):
        #print(ask)
        new_stream=bx.ask('users.txt', stream_size)
        for user in new_stream:
            user_count+=1
            if user_count <= s:
                #print(user_count,len(glob_list))
                glob_list.append(user)
                #print(user_count,len(glob_list))
            else:  
                #print('inside else')
                n=user_count #(ask-1)*s+user_count
                #calculate s/n
                s_over_n = float(s/n)
                floating_num=random.random()
                if floating_num < s_over_n:
                    #remove one user based on the index
                    random_index=random.randint(0,s-1)
                    glob_list[random_index]=user
#                     glob_list.pop(random_index)
#                     glob_list.append(user)            
        #print(ask*s,glob_list[0],glob_list[20],glob_list[40],glob_list[60],glob_list[80]) 
        
        
        final.append("{},{},{},{},{},{}\n".format(ask*s,glob_list[0],glob_list[20],glob_list[40],glob_list[60],glob_list[80]))

    with open(output_file, 'w') as fp:
        fp.write("seqnum,0_id,20_id,40_id,60_id,80_id\n")
        fp.writelines(final)
end=time.time()
print("**********************************************")
print("**********************************************")
print("Duration:",end-start)
print("**********************************************")
print("**********************************************")

if __name__ == '__main__':
    input_file=sys.argv[1]
    output_file=sys.argv[4]
    num_of_asks=int(sys.argv[3])
    stream_size=int(sys.argv[2])

    main(input_file,stream_size,num_of_asks,output_file)
    