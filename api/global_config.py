# global_config.py
from typing import Optional
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

class BrainHeartSettings:
    def __init__(self, brain_provider: Optional[str], brain_model: Optional[str],
                 heart_provider: Optional[str], heart_model: Optional[str],
                 indic_provider: Optional[str], indic_model: Optional[str],
                 routing_provider: Optional[str], routing_model: Optional[str],
                 simple_whatsapp_provider: Optional[str], simple_whatsapp_model: Optional[str],
                 cot_whatsapp_provider: Optional[str], cot_whatsapp_model: Optional[str],
                 use_premium_search: bool, web_model: Optional[str]):
        self.brain_provider = brain_provider
        self.brain_model = brain_model
        self.heart_provider = heart_provider
        self.heart_model = heart_model
        self.indic_provider = indic_provider
        self.indic_model = indic_model
        self.routing_provider = routing_provider
        self.routing_model = routing_model
        self.simple_whatsapp_provider = simple_whatsapp_provider
        self.simple_whatsapp_model = simple_whatsapp_model
        self.cot_whatsapp_provider = cot_whatsapp_provider
        self.cot_whatsapp_model = cot_whatsapp_model
        self.use_premium_search = use_premium_search
        self.web_model = web_model

settings = BrainHeartSettings(
    brain_provider=os.getenv('BRAIN_LLM_PROVIDER'),
    brain_model=os.getenv('BRAIN_LLM_MODEL'),
    heart_provider=os.getenv('HEART_LLM_PROVIDER'),
    heart_model=os.getenv('HEART_LLM_MODEL'),
    indic_provider=os.getenv('INDIC_HEART_LLM_PROVIDER'),
    indic_model=os.getenv('INDIC_HEART_LLM_MODEL'),
    routing_provider=os.getenv('ROUTING_LLM_PROVIDER'),
    routing_model=os.getenv('ROUTING_LLM_MODEL'),
    simple_whatsapp_provider=os.getenv('SIMPLE_WHATSAPP_LLM_PROVIDER'),
    simple_whatsapp_model=os.getenv('SIMPLE_WHATSAPP_LLM_MODEL'),
    cot_whatsapp_provider=os.getenv('COT_WHATSAPP_LLM_PROVIDER'),
    cot_whatsapp_model=os.getenv('COT_WHATSAPP_LLM_MODEL'),
    use_premium_search=os.getenv('USE_PREMIUM_SEARCH', 'false').lower() == 'true',
    web_model=os.getenv('WEB_MODEL', None)
)
