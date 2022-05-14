#!/usr/bin/env python3

"""
STREAM CLASSIFICATION INFERENCE
===============================

The following program is used to perform inference on the subject data
"""

# %%
# Importing Libraries
from model import *
import time
import sys
import argparse
import socket
from concurrent import futures
import grpc
import communication_pb2
import communication_pb2_grpc

# %%
# Main Inference Class
class Inference:
    def __init__(self, OHE, laddr, sc_window=5):
        """
        This method is used to initialize Model inference

        Method Input
        =============
        OHE : Absolute address of one hot encoded labels file
        laddr : Absolute address of the file to which model is subject to load
        sc_window : Stream classification window size to observe

        Method Output
        ==============
        None
        """
        torch.cuda.empty_cache()
        self.OHE_address = OHE
        self.model_address = laddr
        self.__sc_window_size__ = sc_window
        self.__device__ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        with open(self.OHE_address, 'rb') as file1:
            self.class_ohe = pickle.load(file1)
            self.__stream_list__ = list(self.class_ohe.keys())
        self.reverse_class_ohe = { values : keys for keys, values in self.class_ohe.items() }
        print('>>>>> Stream Classification Labels Loaded')
        self.mod_probs = torch.nn.Softmax(dim=1)
        self.mod = Model(len(self.class_ohe))
        self.mod.load_state_dict(torch.load(self.model_address, map_location = self.__device__))
        self.mod.to(self.__device__)
        self.mod.eval()
        print('>>>>> Inference Model Loaded')
        self.__sc_win_data__ = dict()
    
    def __str__(self):
        """
        This method is __str__ implementation of subject class

        Method Input
        =============
        None

        Method Output
        ==============
        New Line
        """
        print(f'Acceleration Device: {self.__device__}')
        print(f'Stream Classification Averaging Window Size: {self.__sc_window_size__}')
        print(f'Number of Stream Classificaion Labels: {len(self.__stream_list__)}')
        print(f'Stream Classificaion Labels: {self.__stream_list__}')
        return '\n'
    
    def __input_batch__(self, btch_dat):
        """
        This method converts stacked Numpy PIL images to stacked Torch tensors

        Method Input
        =============
        btch_dat : Stacked Numpy PIL images with following shape:
                            [ Batch x Width x Height x Channel ]

        Method Output
        ==============
        Torch tensor
        """
        input_tensors = [inference_transforms(Image.fromarray(i)) for i in btch_dat]
        return torch.stack(input_tensors)
    
    def __sc_probs__(self, inf_probs, client_id = 'abcdefghij'):
        """
        This method is used to find average probabilities over samples for subject batch for Stream Classification

        Method Input
        =============
        inf_probs : Stream classification output in the form of Numpy probabilities
        client_id : Client ID to identify connection & track inference ( default: abcdefghij )

        Method Output
        ==============
        Averaged probabilities as Numpy array
        """
        curr_sizer = inf_probs.shape[0] 
        if len(self.__sc_win_data__[client_id]) == 0:
            self.__sc_win_data__[client_id].append(inf_probs)
        else:
            prev_sizer = max([i.shape[0] for i in self.__sc_win_data__[client_id]])
            if curr_sizer <= prev_sizer:
                inf_probs = np.pad(inf_probs, ((0,prev_sizer-curr_sizer),(0,0)))
            else:
                self.__sc_win_data__[client_id] = [np.pad(i, ((0,curr_sizer-prev_sizer),(0,0)))for i in self.__sc_win_data__[client_id]]
            self.__sc_win_data__[client_id].append(inf_probs)
            if len(self.__sc_win_data__[client_id]) > self.__sc_window_size__:
                del self.__sc_win_data__[client_id][0]
        avg = sum(self.__sc_win_data__[client_id]) / len(self.__sc_win_data__[client_id])
        return avg[:curr_sizer]
    
    def __call__(self, img_btch, client_id = 'abcdefghij'):
        """
        This method is used to perform Stream Classification inference

        Method Input
        =============
        img_btch : Stacked Numpy PIL images with following shape:
                            [ Batch x Width x Height x Channel ]
        client_id : Client ID to identify connection & track inference ( default: abcdefghij )

        Method Output
        ==============
        Stream Classification as tuple
                            ( Stream Classification Inference List, Stream Classification Probability List )
        """
        if client_id not in list(self.__sc_win_data__.keys()):
            self.__sc_win_data__[client_id] = list()
        img_btch = self.__input_batch__(img_btch)
        inf1 = self.mod(img_btch.to(self.__device__))
        sc_out1p = self.__sc_probs__(self.mod_probs(inf1).cpu().detach().numpy(), client_id)
        sc_argmax = np.argmax(sc_out1p, axis=1)
        sc_out1 = [self.reverse_class_ohe[i] for i in sc_argmax]
        sc_outp = [j[i] for i, j in zip(sc_argmax, sc_out1p)]
        return sc_out1, np.array(sc_outp)

# %%
# Inference Server Class
class sc_service(communication_pb2_grpc.sc_serviceServicer):
    def __init__(self, *args, **kwargs):
        """
        This method is used to initialize server class for Model inference

        Method Input
        =============
        None

        Method Output
        ==============
        None
        """
        pass

    def __request_processor(self, inp_req):
        """
        This method is used to process input request to server

        Method Input
        =============
        inp_req : Request object generated by GRPC

        Method Output
        ==============
        Input data for inference as Numpy array along with client id
                            ( Input Numpy data, Client ID )
        """
        ret_dat = np.frombuffer(inp_req.imgs, dtype=inp_req.data_type).reshape(inp_req.batch, inp_req.width, inp_req.height, inp_req.channel)
        print(f'Input Batch Shape: {ret_dat.shape}')
        return ret_dat, inp_req.client_id
    
    def __output_processor(self, inf_out):
        """
        This method is used to process output data after inference

        Method Input
        =============
        inf_out : Stream Classification & Brand Recognition inference as tuple
                            ( Stream Classification Inference List, Brand Recognition Inference List )

        Method Output
        ==============
        Output object after inference
        """
        return communication_pb2.server_output(
            stream_classification = np.array(inf_out[0]).tobytes(),
            probabilities = np.array(inf_out[1]).tobytes(),
            data_type = inf_out[1].dtype.name
        )

    def inference(self, request, context):
        """
        This method is used to handle requests & inference outputs

        Method Input
        =============
        request : GPRC generated input request object
        context : GRPC generated API context

        Method Output
        ==============
        None
        """
        inp_data, client_id = self.__request_processor(request)
        print(f'Client ID: {client_id}')
        st = time.time()
        out_data = inf_obj(inp_data, client_id)
        ed = time.time() - st
        print(f'Inference Time: {ed}')
        print(f'Stream Classification: {out_data[0]}')
        print('---------------------------------------------\n')
        return self.__output_processor(out_data)

# %%
# Server Execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Stream Classification Inference Server.')
    parser.add_argument('-scw', '--sc_window', type = int, help = 'Stream Classification Averaging Window Size', default = 5)
    parser.add_argument('-ip', '--server_ip', type = str, help = 'IP Address to Start GRPC Server', default = '[::]:1234')
    parser.add_argument('-msg', '--msg_len', type = int, help = 'Message Length Subject to Communication by GRPC', default = 1000000000)
    parser.add_argument('-wrk', '--workers', type = int, help = 'Number of Workers to Used by GRPC', default = 1)
    parser.add_argument('-ohe', '--OHE', type = str, help = 'Absolute Address of One Hot Encoded Labels File', default = '/resources/OHE.labels')
    parser.add_argument('-la', '--laddr', type = str, help = 'Absolute Address of Model File', default = '/resources/convnext.model')
    args = vars(parser.parse_args())
    print("""
    ==========================================
    | Stream Classification Inference Server |
    ==========================================
    """)
    inf_obj = Inference(OHE = args['OHE'], laddr = args['laddr'], sc_window = args['sc_window'])
    print('---------------------------------------------')
    print(inf_obj)
    print("""
    =============================================
    |       Inference GRPC Server Details       |
    =============================================
    """)
    duip, msle, wrke = args['server_ip'], args['msg_len'], args['workers']
    print(f'Inference IP: {duip}')
    print(f'Server IP: {socket.gethostbyname(socket.gethostname())}')
    print(f'Maximum Server Communication Message Length: {msle}')
    print(f'Number of Worker Allowed for GRPC Server: {wrke}')
    print('---------------------------------------------')
    print('>>>>> Press Ctrl+C To Shutdown Server')
    print("""
    =============================================
    |              Inference Logs               |
    =============================================
    """)
    server_opts = [('grpc.max_send_message_length', args['msg_len']), ('grpc.max_receive_message_length', args['msg_len'])]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers = args['workers']), options = server_opts)
    communication_pb2_grpc.add_sc_serviceServicer_to_server(sc_service(), server)
    server.add_insecure_port(args['server_ip'])
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("""
    =============================================
    |               Shutting Down               |
    =============================================
    """)
        sys.exit(0)
