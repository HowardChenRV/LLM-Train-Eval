#!/usr/bin/env python3
import os
import functools
import json
from retrying import retry
from aurora.common.logger import Logger

logger = Logger()
current_dir = os.path.dirname(os.path.abspath(__file__))


class Config:
    def __init__(self):
        self.bootstrap_servers = os.getenv("AURORA_KAFKA_SERVER", "10.120.1.5:9094")
        self.default_group = "group_infini_tbench"
        self.security_protocol = "SASL_PLAINTEXT"
        self.sasl_mechanism = "PLAIN"
        self.sasl_plain_username = os.getenv("AURORA_KAFKA_SASL_USERNAME", "aurora")
        self.sasl_plain_password = os.getenv("AURORA_KAFKA_SASL_PASSWORD", "Welcome1n!")


@functools.lru_cache(maxsize=1)
def get_config():
    return Config()


@retry(stop_max_attempt_number=5, wait_fixed=2000)
def get_producer():
    if os.getenv("DATA_CLIENT_MODE", "online") != "offline":
        from kafka import  KafkaProducer
        cfg = get_config()
        return KafkaProducer(
            bootstrap_servers=[cfg.bootstrap_servers],
            security_protocol=cfg.security_protocol,
            sasl_mechanism=cfg.sasl_mechanism,
            sasl_plain_username=cfg.sasl_plain_username,
            sasl_plain_password=cfg.sasl_plain_password,
            value_serializer=lambda m: json.dumps(m).encode('utf-8'),
        )
    else:
        # 如果offline, 则不上传数据，只记录本地日志
        logger.warning("envrionment variable DATA_CLIENT_MODE == offline, won't send data to data client!")
        return None