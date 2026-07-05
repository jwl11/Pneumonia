from libs.PipeLine import PipeLine, ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
import os,sys,gc,time,random,utime
import ujson
from media.media import *
from time import *
import nncase_runtime as nn
import ulab.numpy as np
import image
import aidemo
import network
import time
#from simple import MQTTClient  #  直接从根目录的 simple.py 导入
import ustruct as struct
#串口通信
from machine import FPIOA
from machine import Pin
from machine import UART

#wifi

# === MQTT 服务器参数配置 ===
MQTT_SERVER = "broker-cn.emqx.io"  # MQTT 服务器地址（这里用的是 EMQX 免费公共测试平台）
MQTT_PORT   = 1883              # 常规 TCP 连接端口
MQTT_USER   = ""                # 用户名（如果服务器不需要则留空）
MQTT_PWD    = ""                # 密码（如果服务器不需要则留空）
CLIENT_ID   = "K230_Lushan_AI"  # 客户端唯一ID，多台设备千万不能重复
TOPIC_PUB   = "k230/ai_result"  # 发布 AI 结果的话题（主题）

mqtt_client = None

SSID = "wyn"        # 路由器名称
PASSWORD = "00000000" # 路由器密码

def sta_test():
    # 初始化STA模式（客户端模式）
    sta = network.WLAN(network.STA_IF)

    # 激活WiFi模块（相当于打开手机WIFI开关）
    if not sta.active():  # 判断是否已激活
        sta.active(True)
    print("WiFi模块激活状态:", sta.active())

    # 查看初始连接状态
    print("初始连接状态:", sta.status())

    # 扫描当前环境中的WIFI
    wifi_list = sta.scan()  # 扫描周围WiFi
    # 打印每个Wi-Fi信息
    for wifi in wifi_list:
        # 访问 rt_wlan_info 对象的属性
        ssid = wifi.ssid       # ssid 属性
        rssi = wifi.rssi       # rssi 属性
        print(f"SSID: {ssid}, 信号强度: {rssi}dBm")

    # 尝试连接路由器
    print(f"正在连接 {SSID}...")
    sta.connect(SSID, PASSWORD)

    # 等待连接结果（最多尝试5次）
    max_wait = 5
    while max_wait > 0:
        if sta.isconnected():  # 检查是否连接成功
            break
        max_wait -= 1
        time.sleep(1)  # 失败了就线休息一秒再说
        sta.connect(SSID, PASSWORD)
        print("剩余等待次数：", max_wait, "次")

    # 如果获取不到IP地址就一直在这等待
    while sta.ifconfig()[0] == '0.0.0.0':
        pass

    if sta.isconnected():
        print("\n连接成功！")
        # 重新获取并打印网络配置
        ip_info = sta.ifconfig()
        print(f"IP地址: {ip_info[0]}")
        print(f"子网掩码: {ip_info[1]}")
        print(f"网关: {ip_info[2]}")
        print(f"DNS服务器: {ip_info[3]}")
    else:
        print("连接失败，请检查密码或信号强度")

def connect_mqtt():
    global mqtt_client
    print(f"正在尝试使用极简协议连接 MQTT 服务器 [{MQTT_SERVER}]...")
    try:
        import usocket as socket
        # 1. 创建原生 TCP Socket
        sock = socket.socket()
        addr = socket.getaddrinfo(MQTT_SERVER, MQTT_PORT)[0][-1]
        sock.connect(addr)

        # 2. 手动组装 MQTT 3.1.1 核心连接报文（不带用户名密码的干净版本）
        client_id_bytes = CLIENT_ID.encode('utf-8')
        # 固定报文头
        fixed_header = b"\x10"
        # 剩余长度 = 协议名长度(2) + 协议名("MQTT"=4) + 协议版本(1) + 连接标志(1) + 保持连接时间(2) + 客户端ID长度(2) + 客户端ID实际长度
        remaining_len = 12 + len(client_id_bytes)

        # 拼接变量包头
        variable_header = b"\x00\x04MQTT\x04\x02\x00\x3c" # 协议名MQTT, 版本4(3.1.1), 清理会话, KeepAlive=60s
        client_len_bytes = struct.pack("!H", len(client_id_bytes))

        # 组合成完整的 CONNECT 报文并发送
        connect_packet = fixed_header + bytes([remaining_len]) + variable_header + client_len_bytes + client_id_bytes
        sock.write(connect_packet)

        # 3. 稍微等待并读取服务器回应，做安全的越界保护
        time.sleep(0.5)
        resp = sock.read(4)
        if resp and len(resp) >= 4 and resp[0] == 0x20 and resp[3] == 0:
            print("MQTT 服务器连接成功！")
        else:
            print("MQTT 握手异常，但通道已建立，转为盲发模式。")

        # 4. 把这个原生的 socket 伪装赋给 mqtt_client 供 draw_result 使用
        class CustomMQTT:
            def __init__(self, s): self.sock = s
            def publish(self, topic, msg):
                # 动态构建不带认证的极简 QoS 0 发布报文
                topic_bytes = topic.encode('utf-8') if isinstance(topic, str) else topic
                msg_bytes = msg if isinstance(msg, bytes) else msg.encode('utf-8')
                sz = 2 + len(topic_bytes) + len(msg_bytes)
                self.sock.write(bytearray([0x30, sz]))
                self.sock.write(struct.pack("!H", len(topic_bytes)))
                self.sock.write(topic_bytes)
                self.sock.write(msg_bytes)

        mqtt_client = CustomMQTT(sock)
        # 往主题发送一条上线测试文本
        mqtt_client.publish(TOPIC_PUB, b"K230 Native Connected!")
        return True
    except Exception as e:
        print("MQTT 连接失败，错误信息:", e)
        mqtt_client = None
        return False

#wifi_end

# 自定义人脸检测类，继承自AIBase基类
class FaceDetectionApp(AIBase):
    def __init__(self, kmodel_path, model_input_size, anchors, confidence_threshold=0.5, nms_threshold=0.2, rgb888p_size=[224,224], display_size=[1920,1080], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)  # 调用基类的构造函数
        self.kmodel_path = kmodel_path  # 模型文件路径
        self.model_input_size = model_input_size  # 模型输入分辨率
        self.confidence_threshold = confidence_threshold  # 置信度阈值
        self.nms_threshold = nms_threshold  # NMS（非极大值抑制）阈值
        self.anchors = anchors  # 锚点数据，用于目标检测
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]  # sensor给到AI的图像分辨率，并对宽度进行16的对齐
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]  # 显示分辨率，并对宽度进行16的对齐
        self.debug_mode = debug_mode  # 是否开启调试模式
        self.ai2d = Ai2d(debug_mode)  # 实例化Ai2d，用于实现模型预处理
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)  # 设置Ai2d的输入输出格式和类型

    # 配置预处理操作，这里使用了pad和resize，Ai2d支持crop/shift/pad/resize/affine，具体代码请打开/sdcard/app/libs/AI2D.py查看
    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):  # 计时器，如果debug_mode大于0则开启
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size  # 初始化ai2d预处理配置，默认为sensor给到AI的尺寸，可以通过设置input_image_size自行修改输入尺寸
            top, bottom, left, right = self.get_padding_param()  # 获取padding参数
            self.ai2d.pad([0, 0, 0, 0, top, bottom, left, right], 0, [104, 117, 123])  # 填充边缘
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)  # 缩放图像
            self.ai2d.build([1,3,ai2d_input_size[1],ai2d_input_size[0]],[1,3,self.model_input_size[1],self.model_input_size[0]])  # 构建预处理流程

    # 自定义当前任务的后处理，results是模型输出array列表，这里使用了aidemo库的face_det_post_process接口
    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):

            scores = results[0][0]   # (3,)

            class_id = int(np.argmax(scores))
            score = float(scores[class_id])

            return class_id, score
    # 绘制检测结果到画面上
    def draw_result(self, pl, res):
        with ScopedTiming("display_draw", self.debug_mode > 0):

            pl.osd_img.clear()

            if res is None:
                pl.show_image()
                return

            class_id, score = res

            text = "class:{} score:{:.2f}".format(class_id, score)

            pl.osd_img.draw_string_advanced(
                50, 50, 40,
                text,
                color=(255, 0, 0, 255)
            )

            global uart2
            # 将概率乘以 100 并强转为整数（例如 0.89 变成 89）
            score_int = int(score * 100)

            # 限制范围在 0~100 之间，防止意外溢出
            score_int = max(0, min(100, score_int))

            # 构建固定的4字节数据包：[帧头FF, 类型012, 概率%, 帧尾FE]
            full_packet = bytes([0xFF, class_id, score_int, 0xFE])

            # 串口一次性发出 6 个纯字节
            uart2.write(full_packet)

            # 4. === 通过 MQTT 发布出去 ===
            global mqtt_client
            if mqtt_client:
                try:
                    # 将这 4 个字节的数据包发布到指定的话题 "k230/ai_result"
                    mqtt_client.publish(TOPIC_PUB, full_packet)
                except Exception as e:
                    print("MQTT 发送失败，尝试重新连接...", e)
                    connect_mqtt() # 掉线自动重连

            pl.show_image()

    # 获取padding参数
    def get_padding_param(self):
        dst_w = self.model_input_size[0]  # 模型输入宽度
        dst_h = self.model_input_size[1]  # 模型输入高度
        ratio_w = dst_w / self.rgb888p_size[0]  # 宽度缩放比例
        ratio_h = dst_h / self.rgb888p_size[1]  # 高度缩放比例
        ratio = min(ratio_w, ratio_h)  # 取较小的缩放比例
        new_w = int(ratio * self.rgb888p_size[0])  # 新宽度
        new_h = int(ratio * self.rgb888p_size[1])  # 新高度
        dw = (dst_w - new_w) / 2  # 宽度差
        dh = (dst_h - new_h) / 2  # 高度差
        top = int(round(0))
        bottom = int(round(dh * 2 + 0.1))
        left = int(round(0))
        right = int(round(dw * 2 - 0.1))
        return top, bottom, left, right

if __name__ == "__main__":
    # 显示模式，默认"hdmi",可以选择"hdmi"和"lcd"
    display_mode="lcd"
    # k230保持不变，k230d可调整为[640,360]
    rgb888p_size = [640,360]

    if display_mode=="hdmi":
        display_size=[1920,1080]
    else:
        display_size=[800,480]
    # 设置模型路径和其他参数
    kmodel_path = "/sdcard/best.kmodel"
    # 其它参数
    confidence_threshold = 0.7
    nms_threshold = 0.2
    anchor_len = None
    det_dim = 4
    anchors_path = None
    anchors = None
    anchors = None
    # 初始化PipeLine，用于图像处理流程
    pl = PipeLine(rgb888p_size=rgb888p_size, display_size=display_size, display_mode=display_mode)
    pl.create()  # 创建PipeLine实例
    # 初始化自定义人脸检测实例
    yolo_det = FaceDetectionApp(kmodel_path, model_input_size=[320, 320], anchors=anchors, confidence_threshold=confidence_threshold, nms_threshold=nms_threshold, rgb888p_size=rgb888p_size, display_size=display_size, debug_mode=0)
    yolo_det.config_preprocess()  # 配置预处理

    # === usart 串口配置 ===
    fpioa=FPIOA()
    fpioa.set_function(5, FPIOA.UART2_TXD)
    fpioa.set_function(6, FPIOA.UART2_RXD)
    uart2 = UART(UART.UART2, baudrate=115200, bits=8, parity=UART.PARITY_NONE, stop=1)
    #sta_test()
    #connect_mqtt()

    try:
        while True:
            os.exitpoint()                      # 检查是否有退出信号
            with ScopedTiming("total",1):
                img = pl.get_frame()            # 获取当前帧数据
                res = yolo_det.run(img)         # 推理当前帧
                yolo_det.draw_result(pl, res)   # 绘制结果
                #pl.show_image()                 # 显示结果
                gc.collect()                    # 垃圾回收
    except Exception as e:
        print(e)                  # 打印异常信息
    finally:
        yolo_det.deinit()                       # 反初始化
        pl.destroy()                            # 销毁PipeLine实例
