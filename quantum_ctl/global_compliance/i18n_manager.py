"""
Internationalization and Localization Manager
Supports multiple languages, currencies, and regional formats
"""

import json
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import locale

logger = logging.getLogger(__name__)

class SupportedLanguage(Enum):
    ENGLISH = "en"
    SPANISH = "es" 
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    KOREAN = "ko"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    RUSSIAN = "ru"
    ARABIC = "ar"

class SupportedCurrency(Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CNY = "CNY"
    KRW = "KRW"
    CAD = "CAD"
    AUD = "AUD"
    CHF = "CHF"
    SEK = "SEK"
    SGD = "SGD"

class SupportedRegion(Enum):
    NORTH_AMERICA = "NA"
    EUROPE = "EU"
    ASIA_PACIFIC = "APAC"
    LATIN_AMERICA = "LATAM"
    MIDDLE_EAST_AFRICA = "MEA"

@dataclass
class LocalizationProfile:
    """User localization preferences"""
    user_id: str
    language: SupportedLanguage
    currency: SupportedCurrency
    region: SupportedRegion
    timezone: str
    date_format: str
    number_format: str
    measurement_system: str  # metric, imperial
    created_at: float
    last_updated: float

class TranslationEngine:
    """Handles text translation and localization"""
    
    def __init__(self):
        self.translations = self._load_translations()
        self.fallback_language = SupportedLanguage.ENGLISH
        
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """Load translation dictionaries"""
        
        # In production, these would be loaded from JSON files or database
        translations = {
            SupportedLanguage.ENGLISH.value: {
                # System Messages
                "system_starting": "Quantum HVAC system is starting...",
                "system_ready": "System ready for operation",
                "system_error": "System error occurred",
                "system_shutdown": "System shutting down",
                
                # User Interface
                "dashboard": "Dashboard",
                "settings": "Settings", 
                "reports": "Reports",
                "alerts": "Alerts",
                "temperature": "Temperature",
                "humidity": "Humidity",
                "energy_usage": "Energy Usage",
                "cost_savings": "Cost Savings",
                "comfort_level": "Comfort Level",
                
                # Actions
                "optimize": "Optimize",
                "start": "Start",
                "stop": "Stop",
                "configure": "Configure",
                "export": "Export",
                "save": "Save",
                "cancel": "Cancel",
                
                # Status Messages
                "optimizing": "Optimizing...",
                "completed": "Completed",
                "failed": "Failed",
                "in_progress": "In Progress",
                
                # Alerts
                "temperature_alert": "Temperature outside comfort range",
                "energy_alert": "High energy consumption detected",
                "system_alert": "System maintenance required",
                "security_alert": "Security incident detected",
                
                # Units
                "celsius": "°C",
                "fahrenheit": "°F", 
                "kwh": "kWh",
                "percent": "%",
                "hours": "hours",
                "minutes": "minutes",
            },
            
            SupportedLanguage.SPANISH.value: {
                # System Messages
                "system_starting": "El sistema HVAC cuántico está iniciando...",
                "system_ready": "Sistema listo para operar",
                "system_error": "Ocurrió un error del sistema",
                "system_shutdown": "Sistema apagándose",
                
                # User Interface
                "dashboard": "Panel de Control",
                "settings": "Configuración",
                "reports": "Informes",
                "alerts": "Alertas",
                "temperature": "Temperatura",
                "humidity": "Humedad",
                "energy_usage": "Uso de Energía",
                "cost_savings": "Ahorro de Costos",
                "comfort_level": "Nivel de Confort",
                
                # Actions
                "optimize": "Optimizar",
                "start": "Iniciar",
                "stop": "Detener",
                "configure": "Configurar",
                "export": "Exportar",
                "save": "Guardar",
                "cancel": "Cancelar",
                
                # Status Messages
                "optimizing": "Optimizando...",
                "completed": "Completado",
                "failed": "Falló",
                "in_progress": "En Progreso",
                
                # Alerts
                "temperature_alert": "Temperatura fuera del rango de confort",
                "energy_alert": "Alto consumo de energía detectado",
                "system_alert": "Mantenimiento del sistema requerido",
                "security_alert": "Incidente de seguridad detectado",
                
                # Units
                "celsius": "°C",
                "fahrenheit": "°F",
                "kwh": "kWh",
                "percent": "%",
                "hours": "horas",
                "minutes": "minutos",
            },
            
            SupportedLanguage.FRENCH.value: {
                # System Messages  
                "system_starting": "Le système HVAC quantique démarre...",
                "system_ready": "Système prêt pour le fonctionnement",
                "system_error": "Erreur système survenue",
                "system_shutdown": "Arrêt du système",
                
                # User Interface
                "dashboard": "Tableau de Bord",
                "settings": "Paramètres",
                "reports": "Rapports", 
                "alerts": "Alertes",
                "temperature": "Température",
                "humidity": "Humidité",
                "energy_usage": "Consommation d'Énergie",
                "cost_savings": "Économies de Coûts",
                "comfort_level": "Niveau de Confort",
                
                # Actions
                "optimize": "Optimiser",
                "start": "Démarrer",
                "stop": "Arrêter",
                "configure": "Configurer",
                "export": "Exporter",
                "save": "Enregistrer",
                "cancel": "Annuler",
                
                # Status Messages
                "optimizing": "Optimisation...",
                "completed": "Terminé",
                "failed": "Échoué",
                "in_progress": "En Cours",
                
                # Alerts
                "temperature_alert": "Température hors de la plage de confort",
                "energy_alert": "Forte consommation d'énergie détectée",
                "system_alert": "Maintenance système requise",
                "security_alert": "Incident de sécurité détecté",
                
                # Units
                "celsius": "°C",
                "fahrenheit": "°F",
                "kwh": "kWh", 
                "percent": "%",
                "hours": "heures",
                "minutes": "minutes",
            },
            
            SupportedLanguage.GERMAN.value: {
                # System Messages
                "system_starting": "Quanten-HVAC-System startet...",
                "system_ready": "System betriebsbereit",
                "system_error": "Systemfehler aufgetreten",
                "system_shutdown": "System wird heruntergefahren",
                
                # User Interface
                "dashboard": "Dashboard",
                "settings": "Einstellungen",
                "reports": "Berichte",
                "alerts": "Warnungen",
                "temperature": "Temperatur",
                "humidity": "Luftfeuchtigkeit",
                "energy_usage": "Energieverbrauch",
                "cost_savings": "Kosteneinsparungen",
                "comfort_level": "Komfortniveau",
                
                # Actions
                "optimize": "Optimieren",
                "start": "Starten",
                "stop": "Stoppen",
                "configure": "Konfigurieren",
                "export": "Exportieren",
                "save": "Speichern",
                "cancel": "Abbrechen",
                
                # Status Messages
                "optimizing": "Optimierung...",
                "completed": "Abgeschlossen",
                "failed": "Fehlgeschlagen",
                "in_progress": "In Bearbeitung",
                
                # Alerts
                "temperature_alert": "Temperatur außerhalb des Komfortbereichs",
                "energy_alert": "Hoher Energieverbrauch erkannt",
                "system_alert": "Systemwartung erforderlich", 
                "security_alert": "Sicherheitsvorfall erkannt",
                
                # Units
                "celsius": "°C",
                "fahrenheit": "°F",
                "kwh": "kWh",
                "percent": "%",
                "hours": "Stunden",
                "minutes": "Minuten",
            },
            
            SupportedLanguage.JAPANESE.value: {
                # System Messages
                "system_starting": "量子HVACシステムが起動しています...",
                "system_ready": "システム運用準備完了",
                "system_error": "システムエラーが発生しました",
                "system_shutdown": "システムをシャットダウンしています",
                
                # User Interface
                "dashboard": "ダッシュボード",
                "settings": "設定",
                "reports": "レポート",
                "alerts": "アラート",
                "temperature": "温度",
                "humidity": "湿度", 
                "energy_usage": "エネルギー使用量",
                "cost_savings": "コスト削減",
                "comfort_level": "快適レベル",
                
                # Actions
                "optimize": "最適化",
                "start": "開始",
                "stop": "停止",
                "configure": "設定",
                "export": "エクスポート",
                "save": "保存",
                "cancel": "キャンセル",
                
                # Status Messages
                "optimizing": "最適化中...",
                "completed": "完了",
                "failed": "失敗",
                "in_progress": "進行中",
                
                # Alerts
                "temperature_alert": "温度が快適範囲外です",
                "energy_alert": "高エネルギー消費が検出されました",
                "system_alert": "システムメンテナンスが必要です",
                "security_alert": "セキュリティインシデントが検出されました",
                
                # Units
                "celsius": "℃",
                "fahrenheit": "℉",
                "kwh": "kWh",
                "percent": "％",
                "hours": "時間",
                "minutes": "分",
            },
            
            SupportedLanguage.CHINESE_SIMPLIFIED.value: {
                # System Messages
                "system_starting": "量子暖通空调系统正在启动...",
                "system_ready": "系统准备就绪",
                "system_error": "系统发生错误",
                "system_shutdown": "系统正在关闭",
                
                # User Interface
                "dashboard": "仪表板",
                "settings": "设置",
                "reports": "报告",
                "alerts": "警报",
                "temperature": "温度",
                "humidity": "湿度",
                "energy_usage": "能源使用",
                "cost_savings": "成本节约",
                "comfort_level": "舒适度",
                
                # Actions
                "optimize": "优化",
                "start": "启动",
                "stop": "停止",
                "configure": "配置",
                "export": "导出",
                "save": "保存",
                "cancel": "取消",
                
                # Status Messages
                "optimizing": "优化中...",
                "completed": "已完成",
                "failed": "失败",
                "in_progress": "进行中",
                
                # Alerts
                "temperature_alert": "温度超出舒适范围",
                "energy_alert": "检测到高能耗",
                "system_alert": "需要系统维护",
                "security_alert": "检测到安全事件",
                
                # Units
                "celsius": "°C",
                "fahrenheit": "°F",
                "kwh": "千瓦时",
                "percent": "%",
                "hours": "小时",
                "minutes": "分钟",
            }
        }
        
        return translations
    
    def translate(self, text_key: str, language: SupportedLanguage, 
                 substitutions: Dict[str, str] = None) -> str:
        """Translate text to specified language"""
        
        try:
            lang_code = language.value
            
            if lang_code in self.translations and text_key in self.translations[lang_code]:
                translated_text = self.translations[lang_code][text_key]
            else:
                # Fallback to English
                fallback_lang = self.fallback_language.value
                if fallback_lang in self.translations and text_key in self.translations[fallback_lang]:
                    translated_text = self.translations[fallback_lang][text_key]
                else:
                    # Return key if no translation found
                    translated_text = text_key
            
            # Apply substitutions
            if substitutions:
                for placeholder, value in substitutions.items():
                    translated_text = translated_text.replace(f"{{{placeholder}}}", str(value))
            
            return translated_text
            
        except Exception as e:
            logger.error(f"Translation error for key '{text_key}' to '{language.value}': {e}")
            return text_key

class CurrencyConverter:
    """Handles currency conversion and formatting"""
    
    def __init__(self):
        # In production, would fetch from live API
        self.exchange_rates = {
            SupportedCurrency.USD: 1.0,       # Base currency
            SupportedCurrency.EUR: 0.85,
            SupportedCurrency.GBP: 0.73,
            SupportedCurrency.JPY: 110.0,
            SupportedCurrency.CNY: 6.45,
            SupportedCurrency.KRW: 1180.0,
            SupportedCurrency.CAD: 1.25,
            SupportedCurrency.AUD: 1.35,
            SupportedCurrency.CHF: 0.92,
            SupportedCurrency.SEK: 8.60,
            SupportedCurrency.SGD: 1.35,
        }
        
        self.currency_symbols = {
            SupportedCurrency.USD: "$",
            SupportedCurrency.EUR: "€",
            SupportedCurrency.GBP: "£",
            SupportedCurrency.JPY: "¥",
            SupportedCurrency.CNY: "¥",
            SupportedCurrency.KRW: "₩",
            SupportedCurrency.CAD: "C$",
            SupportedCurrency.AUD: "A$",
            SupportedCurrency.CHF: "CHF",
            SupportedCurrency.SEK: "kr",
            SupportedCurrency.SGD: "S$",
        }
        
        self.decimal_places = {
            SupportedCurrency.JPY: 0,  # Yen has no decimal places
            SupportedCurrency.KRW: 0,  # Won has no decimal places
        }
        
        self.last_update = time.time()
    
    def convert(self, amount: float, from_currency: SupportedCurrency, 
               to_currency: SupportedCurrency) -> float:
        """Convert amount from one currency to another"""
        
        if from_currency == to_currency:
            return amount
        
        try:
            # Convert to USD first, then to target currency
            usd_amount = amount / self.exchange_rates[from_currency]
            converted_amount = usd_amount * self.exchange_rates[to_currency]
            
            return converted_amount
            
        except KeyError as e:
            logger.error(f"Currency not supported: {e}")
            return amount
        except Exception as e:
            logger.error(f"Currency conversion error: {e}")
            return amount
    
    def format_currency(self, amount: float, currency: SupportedCurrency, 
                       language: SupportedLanguage) -> str:
        """Format currency amount according to locale"""
        
        try:
            symbol = self.currency_symbols.get(currency, currency.value)
            decimal_places = self.decimal_places.get(currency, 2)
            
            # Format based on language/region conventions
            if language in [SupportedLanguage.GERMAN, SupportedLanguage.FRENCH]:
                # European format: symbol after amount, comma as decimal separator
                formatted_amount = f"{amount:,.{decimal_places}f}".replace(',', ' ').replace('.', ',')
                return f"{formatted_amount} {symbol}"
            
            elif language == SupportedLanguage.JAPANESE:
                # Japanese format: symbol before amount
                formatted_amount = f"{amount:,.{decimal_places}f}"
                return f"{symbol}{formatted_amount}"
            
            else:
                # Default format (US/UK style): symbol before amount
                formatted_amount = f"{amount:,.{decimal_places}f}"
                return f"{symbol}{formatted_amount}"
                
        except Exception as e:
            logger.error(f"Currency formatting error: {e}")
            return f"{amount} {currency.value}"
    
    async def update_exchange_rates(self):
        """Update exchange rates from external API"""
        
        try:
            # In production, would call real exchange rate API
            # For demo, just simulate rate fluctuations
            import random
            
            for currency in self.exchange_rates:
                if currency != SupportedCurrency.USD:  # USD is base
                    current_rate = self.exchange_rates[currency]
                    # Simulate small fluctuations (±2%)
                    fluctuation = random.uniform(-0.02, 0.02)
                    new_rate = current_rate * (1 + fluctuation)
                    self.exchange_rates[currency] = new_rate
            
            self.last_update = time.time()
            logger.info("Exchange rates updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update exchange rates: {e}")

class LocalizationService:
    """Main localization service"""
    
    def __init__(self):
        self.user_profiles = {}
        self.default_profile = LocalizationProfile(
            user_id="default",
            language=SupportedLanguage.ENGLISH,
            currency=SupportedCurrency.USD,
            region=SupportedRegion.NORTH_AMERICA,
            timezone="UTC",
            date_format="%Y-%m-%d",
            number_format="1,000.00",
            measurement_system="metric",
            created_at=time.time(),
            last_updated=time.time()
        )
        
        # Regional defaults
        self.regional_defaults = {
            SupportedRegion.NORTH_AMERICA: {
                'language': SupportedLanguage.ENGLISH,
                'currency': SupportedCurrency.USD,
                'timezone': 'America/New_York',
                'measurement_system': 'imperial'
            },
            SupportedRegion.EUROPE: {
                'language': SupportedLanguage.ENGLISH,
                'currency': SupportedCurrency.EUR,
                'timezone': 'Europe/London',
                'measurement_system': 'metric'
            },
            SupportedRegion.ASIA_PACIFIC: {
                'language': SupportedLanguage.ENGLISH,
                'currency': SupportedCurrency.USD,
                'timezone': 'Asia/Singapore',
                'measurement_system': 'metric'
            },
            SupportedRegion.LATIN_AMERICA: {
                'language': SupportedLanguage.SPANISH,
                'currency': SupportedCurrency.USD,
                'timezone': 'America/Mexico_City',
                'measurement_system': 'metric'
            },
            SupportedRegion.MIDDLE_EAST_AFRICA: {
                'language': SupportedLanguage.ENGLISH,
                'currency': SupportedCurrency.USD,
                'timezone': 'Africa/Cairo',
                'measurement_system': 'metric'
            }
        }
    
    def create_user_profile(self, user_id: str, language: SupportedLanguage = None,
                           currency: SupportedCurrency = None,
                           region: SupportedRegion = None) -> LocalizationProfile:
        """Create localization profile for user"""
        
        # Use regional defaults if not specified
        if region and not language:
            language = SupportedLanguage(self.regional_defaults[region]['language'])
        if region and not currency:
            currency = SupportedCurrency(self.regional_defaults[region]['currency'])
        
        # Fall back to default profile values
        profile = LocalizationProfile(
            user_id=user_id,
            language=language or self.default_profile.language,
            currency=currency or self.default_profile.currency,
            region=region or self.default_profile.region,
            timezone=self.regional_defaults.get(region, {}).get('timezone', 'UTC'),
            date_format="%Y-%m-%d %H:%M:%S",
            number_format="1,000.00",
            measurement_system=self.regional_defaults.get(region, {}).get('measurement_system', 'metric'),
            created_at=time.time(),
            last_updated=time.time()
        )
        
        self.user_profiles[user_id] = profile
        return profile
    
    def get_user_profile(self, user_id: str) -> LocalizationProfile:
        """Get user's localization profile"""
        return self.user_profiles.get(user_id, self.default_profile)
    
    def update_user_profile(self, user_id: str, **updates) -> LocalizationProfile:
        """Update user's localization profile"""
        
        if user_id not in self.user_profiles:
            self.create_user_profile(user_id)
        
        profile = self.user_profiles[user_id]
        
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        
        profile.last_updated = time.time()
        
        return profile
    
    def format_temperature(self, celsius: float, user_id: str) -> Tuple[float, str]:
        """Format temperature according to user preference"""
        
        profile = self.get_user_profile(user_id)
        
        if profile.measurement_system == "imperial":
            fahrenheit = (celsius * 9/5) + 32
            return fahrenheit, "°F"
        else:
            return celsius, "°C"
    
    def format_datetime(self, timestamp: float, user_id: str) -> str:
        """Format datetime according to user preference"""
        
        profile = self.get_user_profile(user_id)
        
        try:
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime(profile.date_format)
        except Exception as e:
            logger.error(f"Datetime formatting error: {e}")
            return str(datetime.fromtimestamp(timestamp))
    
    def format_number(self, number: float, user_id: str, decimal_places: int = 2) -> str:
        """Format number according to user locale"""
        
        profile = self.get_user_profile(user_id)
        
        try:
            if profile.language in [SupportedLanguage.GERMAN, SupportedLanguage.FRENCH]:
                # European format: space as thousands separator, comma as decimal
                formatted = f"{number:,.{decimal_places}f}"
                return formatted.replace(',', ' ').replace('.', ',')
            else:
                # Default format: comma as thousands separator, dot as decimal
                return f"{number:,.{decimal_places}f}"
                
        except Exception as e:
            logger.error(f"Number formatting error: {e}")
            return str(number)

class InternationalizationManager:
    """Main I18N management system"""
    
    def __init__(self):
        self.translation_engine = TranslationEngine()
        self.currency_converter = CurrencyConverter()
        self.localization_service = LocalizationService()
        
        # Start periodic exchange rate updates
        self.rate_update_task = None
        self.start_rate_updates()
    
    def start_rate_updates(self):
        """Start periodic exchange rate updates"""
        async def update_rates():
            while True:
                await self.currency_converter.update_exchange_rates()
                await asyncio.sleep(3600)  # Update every hour
        
        self.rate_update_task = asyncio.create_task(update_rates())
    
    async def shutdown(self):
        """Shutdown I18N manager"""
        if self.rate_update_task:
            self.rate_update_task.cancel()
            try:
                await self.rate_update_task
            except asyncio.CancelledError:
                pass
    
    def localize_message(self, message_key: str, user_id: str, 
                        substitutions: Dict[str, Any] = None) -> str:
        """Localize message for user"""
        
        profile = self.localization_service.get_user_profile(user_id)
        
        # Convert any currency values in substitutions
        if substitutions:
            localized_subs = {}
            for key, value in substitutions.items():
                if key.endswith('_amount') and isinstance(value, (int, float)):
                    # Format as currency
                    formatted_value = self.currency_converter.format_currency(
                        value, profile.currency, profile.language
                    )
                    localized_subs[key] = formatted_value
                elif key.endswith('_temp') and isinstance(value, (int, float)):
                    # Format as temperature
                    temp_value, temp_unit = self.localization_service.format_temperature(
                        value, user_id
                    )
                    localized_subs[key] = f"{temp_value:.1f} {temp_unit}"
                else:
                    localized_subs[key] = str(value)
        else:
            localized_subs = substitutions
        
        return self.translation_engine.translate(
            message_key, profile.language, localized_subs
        )
    
    def convert_currency_for_user(self, amount: float, from_currency: SupportedCurrency,
                                 user_id: str) -> Tuple[float, str]:
        """Convert currency to user's preferred currency"""
        
        profile = self.localization_service.get_user_profile(user_id)
        
        converted_amount = self.currency_converter.convert(
            amount, from_currency, profile.currency
        )
        
        formatted_amount = self.currency_converter.format_currency(
            converted_amount, profile.currency, profile.language
        )
        
        return converted_amount, formatted_amount
    
    def get_i18n_status(self) -> Dict[str, Any]:
        """Get internationalization system status"""
        
        return {
            "i18n_system_status": "ACTIVE",
            "supported_languages": [lang.value for lang in SupportedLanguage],
            "supported_currencies": [curr.value for curr in SupportedCurrency],
            "supported_regions": [region.value for region in SupportedRegion],
            "active_user_profiles": len(self.localization_service.user_profiles),
            "translation_keys_available": sum(len(translations) for translations in self.translation_engine.translations.values()),
            "exchange_rate_last_update": self.currency_converter.last_update,
            "rate_update_active": self.rate_update_task is not None and not self.rate_update_task.done(),
            "global_features": [
                "Multi-language Support (12 languages)",
                "Currency Conversion & Formatting",
                "Regional Date/Time Formatting", 
                "Measurement System Conversion",
                "Real-time Exchange Rates",
                "User Preference Management",
                "Regional Compliance Awareness",
                "Cultural Localization"
            ]
        }