"""
Internationalization (i18n) support for quantum HVAC system.
"""

import json
import logging
import os
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class I18nManager:
    """Internationalization manager for multi-language support."""
    
    def __init__(self, default_locale: str = "en"):
        self.default_locale = default_locale
        self.current_locale = default_locale
        self._translations: Dict[str, Dict[str, str]] = {}
        self._load_translations()
    
    def _load_translations(self):
        """Load translation files."""
        # Define translations directly for now
        self._translations = {
            "en": {
                # System messages
                "system.startup": "Quantum HVAC Control System Starting",
                "system.shutdown": "System Shutting Down",
                "system.error": "System Error",
                "system.warning": "System Warning",
                "system.info": "System Information",
                
                # Optimization messages
                "optimization.started": "Optimization Started",
                "optimization.completed": "Optimization Completed Successfully",
                "optimization.failed": "Optimization Failed",
                "optimization.fallback": "Using Fallback Classical Solver",
                "optimization.cache_hit": "Cache Hit - Using Cached Solution",
                "optimization.cache_miss": "Cache Miss - Computing New Solution",
                
                # Building messages
                "building.created": "Building Created",
                "building.validated": "Building Configuration Validated",
                "building.invalid": "Invalid Building Configuration",
                "building.zones": "Zones",
                "building.temperature": "Temperature",
                "building.humidity": "Humidity",
                "building.occupancy": "Occupancy",
                "building.power": "Power Consumption",
                
                # Energy messages
                "energy.savings": "Energy Savings",
                "energy.cost": "Energy Cost",
                "energy.efficiency": "Energy Efficiency",
                "energy.consumption": "Energy Consumption",
                "energy.forecast": "Energy Forecast",
                
                # Units
                "units.celsius": "°C",
                "units.fahrenheit": "°F",
                "units.kilowatt": "kW",
                "units.kilowatthour": "kWh",
                "units.percent": "%",
                "units.dollars": "$",
                "units.euros": "€",
                
                # Status messages
                "status.healthy": "Healthy",
                "status.degraded": "Degraded",
                "status.critical": "Critical",
                "status.available": "Available",
                "status.unavailable": "Unavailable",
                "status.active": "Active",
                "status.inactive": "Inactive",
                
                # Error messages
                "error.validation": "Validation Error",
                "error.connection": "Connection Error",
                "error.timeout": "Timeout Error",
                "error.permission": "Permission Error",
                "error.configuration": "Configuration Error",
                "error.quantum_solver": "Quantum Solver Error",
                
                # UI messages
                "ui.loading": "Loading...",
                "ui.processing": "Processing...",
                "ui.complete": "Complete",
                "ui.cancel": "Cancel",
                "ui.retry": "Retry",
                "ui.close": "Close",
                "ui.save": "Save",
                "ui.export": "Export",
                
                # Reports
                "report.performance": "Performance Report",
                "report.health": "Health Report",
                "report.energy": "Energy Report",
                "report.generated": "Report Generated",
                "report.exported": "Report Exported",
            },
            
            "es": {
                # Sistema
                "system.startup": "Sistema de Control HVAC Cuántico Iniciando",
                "system.shutdown": "Sistema Cerrando",
                "system.error": "Error del Sistema",
                "system.warning": "Advertencia del Sistema", 
                "system.info": "Información del Sistema",
                
                # Optimización
                "optimization.started": "Optimización Iniciada",
                "optimization.completed": "Optimización Completada Exitosamente",
                "optimization.failed": "Optimización Falló",
                "optimization.fallback": "Usando Solver Clásico de Respaldo",
                "optimization.cache_hit": "Cache Hit - Usando Solución en Cache",
                "optimization.cache_miss": "Cache Miss - Calculando Nueva Solución",
                
                # Edificio
                "building.created": "Edificio Creado",
                "building.validated": "Configuración del Edificio Validada",
                "building.invalid": "Configuración de Edificio Inválida",
                "building.zones": "Zonas",
                "building.temperature": "Temperatura",
                "building.humidity": "Humedad",
                "building.occupancy": "Ocupación",
                "building.power": "Consumo de Energía",
                
                # Energía
                "energy.savings": "Ahorro de Energía",
                "energy.cost": "Costo de Energía",
                "energy.efficiency": "Eficiencia Energética",
                "energy.consumption": "Consumo de Energía",
                "energy.forecast": "Pronóstico Energético",
                
                # Unidades
                "units.celsius": "°C",
                "units.fahrenheit": "°F",
                "units.kilowatt": "kW",
                "units.kilowatthour": "kWh",
                "units.percent": "%",
                "units.dollars": "$",
                "units.euros": "€",
                
                # Estado
                "status.healthy": "Saludable",
                "status.degraded": "Degradado",
                "status.critical": "Crítico",
                "status.available": "Disponible",
                "status.unavailable": "No Disponible",
                "status.active": "Activo",
                "status.inactive": "Inactivo",
                
                # Errores
                "error.validation": "Error de Validación",
                "error.connection": "Error de Conexión",
                "error.timeout": "Error de Tiempo de Espera",
                "error.permission": "Error de Permisos",
                "error.configuration": "Error de Configuración",
                "error.quantum_solver": "Error del Solver Cuántico",
                
                # UI
                "ui.loading": "Cargando...",
                "ui.processing": "Procesando...",
                "ui.complete": "Completo",
                "ui.cancel": "Cancelar",
                "ui.retry": "Reintentar",
                "ui.close": "Cerrar",
                "ui.save": "Guardar",
                "ui.export": "Exportar",
                
                # Reportes
                "report.performance": "Reporte de Rendimiento",
                "report.health": "Reporte de Salud",
                "report.energy": "Reporte de Energía",
                "report.generated": "Reporte Generado",
                "report.exported": "Reporte Exportado",
            },
            
            "de": {
                # System
                "system.startup": "Quantum HVAC Steuerungssystem Startet",
                "system.shutdown": "System Wird Heruntergefahren",
                "system.error": "Systemfehler",
                "system.warning": "Systemwarnung",
                "system.info": "Systeminformation",
                
                # Optimierung
                "optimization.started": "Optimierung Gestartet",
                "optimization.completed": "Optimierung Erfolgreich Abgeschlossen",
                "optimization.failed": "Optimierung Fehlgeschlagen",
                "optimization.fallback": "Verwende Klassischen Fallback-Solver",
                "optimization.cache_hit": "Cache-Treffer - Verwende Zwischengespeicherte Lösung",
                "optimization.cache_miss": "Cache-Fehlschlag - Berechne Neue Lösung",
                
                # Gebäude
                "building.created": "Gebäude Erstellt",
                "building.validated": "Gebäudekonfiguration Validiert",
                "building.invalid": "Ungültige Gebäudekonfiguration",
                "building.zones": "Zonen",
                "building.temperature": "Temperatur",
                "building.humidity": "Luftfeuchtigkeit",
                "building.occupancy": "Belegung",
                "building.power": "Energieverbrauch",
                
                # Energie
                "energy.savings": "Energieeinsparungen",
                "energy.cost": "Energiekosten",
                "energy.efficiency": "Energieeffizienz",
                "energy.consumption": "Energieverbrauch",
                "energy.forecast": "Energieprognose",
                
                # Einheiten
                "units.celsius": "°C",
                "units.fahrenheit": "°F",
                "units.kilowatt": "kW",
                "units.kilowatthour": "kWh",
                "units.percent": "%",
                "units.dollars": "$",
                "units.euros": "€",
                
                # Status
                "status.healthy": "Gesund",
                "status.degraded": "Verschlechtert",
                "status.critical": "Kritisch",
                "status.available": "Verfügbar",
                "status.unavailable": "Nicht Verfügbar",
                "status.active": "Aktiv",
                "status.inactive": "Inaktiv",
                
                # Fehler
                "error.validation": "Validierungsfehler",
                "error.connection": "Verbindungsfehler",
                "error.timeout": "Timeout-Fehler",
                "error.permission": "Berechtigungsfehler",
                "error.configuration": "Konfigurationsfehler",
                "error.quantum_solver": "Quantum-Solver-Fehler",
                
                # UI
                "ui.loading": "Laden...",
                "ui.processing": "Verarbeitung...",
                "ui.complete": "Vollständig",
                "ui.cancel": "Abbrechen",
                "ui.retry": "Wiederholen",
                "ui.close": "Schließen",
                "ui.save": "Speichern",
                "ui.export": "Exportieren",
                
                # Berichte
                "report.performance": "Leistungsbericht",
                "report.health": "Gesundheitsbericht",
                "report.energy": "Energiebericht",
                "report.generated": "Bericht Generiert",
                "report.exported": "Bericht Exportiert",
            },
            
            "fr": {
                # Système
                "system.startup": "Système de Contrôle HVAC Quantique Démarrage",
                "system.shutdown": "Arrêt du Système",
                "system.error": "Erreur Système",
                "system.warning": "Avertissement Système",
                "system.info": "Information Système",
                
                # Optimisation
                "optimization.started": "Optimisation Démarrée",
                "optimization.completed": "Optimisation Terminée avec Succès",
                "optimization.failed": "Échec de l'Optimisation",
                "optimization.fallback": "Utilisation du Solveur Classique de Secours",
                "optimization.cache_hit": "Cache Hit - Utilisation de la Solution en Cache",
                "optimization.cache_miss": "Cache Miss - Calcul d'une Nouvelle Solution",
                
                # Bâtiment
                "building.created": "Bâtiment Créé",
                "building.validated": "Configuration du Bâtiment Validée",
                "building.invalid": "Configuration de Bâtiment Invalide",
                "building.zones": "Zones",
                "building.temperature": "Température",
                "building.humidity": "Humidité",
                "building.occupancy": "Occupation",
                "building.power": "Consommation d'Énergie",
                
                # Énergie
                "energy.savings": "Économies d'Énergie",
                "energy.cost": "Coût de l'Énergie",
                "energy.efficiency": "Efficacité Énergétique",
                "energy.consumption": "Consommation d'Énergie",
                "energy.forecast": "Prévision Énergétique",
                
                # Unités
                "units.celsius": "°C",
                "units.fahrenheit": "°F",
                "units.kilowatt": "kW",
                "units.kilowatthour": "kWh",
                "units.percent": "%",
                "units.dollars": "$",
                "units.euros": "€",
                
                # Statut
                "status.healthy": "Sain",
                "status.degraded": "Dégradé",
                "status.critical": "Critique",
                "status.available": "Disponible",
                "status.unavailable": "Indisponible",
                "status.active": "Actif",
                "status.inactive": "Inactif",
                
                # Erreurs
                "error.validation": "Erreur de Validation",
                "error.connection": "Erreur de Connexion",
                "error.timeout": "Erreur de Délai d'Attente",
                "error.permission": "Erreur de Permission",
                "error.configuration": "Erreur de Configuration",
                "error.quantum_solver": "Erreur du Solveur Quantique",
                
                # UI
                "ui.loading": "Chargement...",
                "ui.processing": "Traitement...",
                "ui.complete": "Terminé",
                "ui.cancel": "Annuler",
                "ui.retry": "Réessayer",
                "ui.close": "Fermer",
                "ui.save": "Sauvegarder",
                "ui.export": "Exporter",
                
                # Rapports
                "report.performance": "Rapport de Performance",
                "report.health": "Rapport de Santé",
                "report.energy": "Rapport d'Énergie",
                "report.generated": "Rapport Généré",
                "report.exported": "Rapport Exporté",
            }
        }
        
        logger.info(f"Loaded translations for {len(self._translations)} locales")
    
    def set_locale(self, locale: str):
        """Set current locale."""
        if locale in self._translations:
            self.current_locale = locale
            logger.info(f"Locale set to: {locale}")
        else:
            logger.warning(f"Locale '{locale}' not available, using default: {self.default_locale}")
            self.current_locale = self.default_locale
    
    def get_text(self, key: str, locale: Optional[str] = None, **kwargs) -> str:
        """Get translated text for a key."""
        target_locale = locale or self.current_locale
        
        # Try target locale first
        if target_locale in self._translations:
            translations = self._translations[target_locale]
            if key in translations:
                text = translations[key]
                try:
                    return text.format(**kwargs) if kwargs else text
                except KeyError as e:
                    logger.warning(f"Missing format parameter {e} for key '{key}'")
                    return text
        
        # Fallback to default locale
        if self.default_locale in self._translations:
            translations = self._translations[self.default_locale]
            if key in translations:
                text = translations[key]
                try:
                    return text.format(**kwargs) if kwargs else text
                except KeyError as e:
                    logger.warning(f"Missing format parameter {e} for key '{key}'")
                    return text
        
        # Last resort: return the key itself
        logger.warning(f"Translation key '{key}' not found for locale '{target_locale}'")
        return key
    
    def get_available_locales(self) -> list:
        """Get list of available locales."""
        return list(self._translations.keys())
    
    def format_number(self, number: float, locale: Optional[str] = None) -> str:
        """Format number according to locale conventions."""
        target_locale = locale or self.current_locale
        
        # Simple locale-based number formatting
        if target_locale in ["en"]:
            return f"{number:,.2f}"
        elif target_locale in ["de"]:
            return f"{number:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
        elif target_locale in ["fr", "es"]:
            return f"{number:,.2f}".replace(",", " ").replace(".", ",")
        else:
            return f"{number:.2f}"
    
    def format_currency(self, amount: float, currency: str = "USD", locale: Optional[str] = None) -> str:
        """Format currency according to locale conventions."""
        target_locale = locale or self.current_locale
        formatted_amount = self.format_number(amount, locale)
        
        currency_symbols = {
            "USD": "$",
            "EUR": "€",
            "GBP": "£",
            "JPY": "¥"
        }
        
        symbol = currency_symbols.get(currency, currency)
        
        # Different currency formatting by locale
        if target_locale in ["en"] and currency == "USD":
            return f"${formatted_amount}"
        elif target_locale in ["de", "fr"] and currency == "EUR":
            return f"{formatted_amount} €"
        else:
            return f"{symbol}{formatted_amount}"

# Global i18n manager instance
_i18n_manager = None

def get_i18n_manager() -> I18nManager:
    """Get global i18n manager instance."""
    global _i18n_manager
    if _i18n_manager is None:
        _i18n_manager = I18nManager()
        
        # Auto-detect locale from environment
        import locale as sys_locale
        try:
            system_locale = sys_locale.getdefaultlocale()[0]
            if system_locale:
                lang_code = system_locale.split('_')[0].lower()
                if lang_code in _i18n_manager.get_available_locales():
                    _i18n_manager.set_locale(lang_code)
        except:
            pass  # Use default locale
    
    return _i18n_manager

def _(key: str, **kwargs) -> str:
    """Convenience function for getting translated text."""
    return get_i18n_manager().get_text(key, **kwargs)