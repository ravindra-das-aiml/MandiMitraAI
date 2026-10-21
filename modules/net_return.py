"""Net-return calculator for comparing mandis (markets).

Estimated net return = gross sale - transport - handling - commission
All amounts are in Indian rupees (INR). Prices are per quintal.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class MarketOption:
    name: str
    price_per_quintal: float
    transport_cost: float = 0.0
    handling_cost_per_quintal: float = 0.0
    commission_percent: float = 0.0


@dataclass(frozen=True)
class NetReturnResult:
    market: str
    quantity_quintals: float
    price_per_quintal: float
    gross_sale: float
    transport_cost: float
    handling_cost: float
    commission: float
    total_costs: float
    net_return: float


def calculate_net_return(quantity_quintals, option):
    if quantity_quintals <= 0:
        raise ValueError("Quantity must be greater than zero.")
    if option.price_per_quintal < 0:
        raise ValueError("Price cannot be negative.")
    if option.transport_cost < 0 or option.handling_cost_per_quintal < 0:
        raise ValueError("Costs cannot be negative.")
    if not 0 <= option.commission_percent <= 100:
        raise ValueError("Commission percent must be between 0 and 100.")

    gross_sale = quantity_quintals * option.price_per_quintal
    handling_cost = quantity_quintals * option.handling_cost_per_quintal
    commission = gross_sale * option.commission_percent / 100
    total_costs = option.transport_cost + handling_cost + commission

    return NetReturnResult(
        market=option.name,
        quantity_quintals=quantity_quintals,
        price_per_quintal=option.price_per_quintal,
        gross_sale=gross_sale,
        transport_cost=option.transport_cost,
        handling_cost=handling_cost,
        commission=commission,
        total_costs=total_costs,
        net_return=gross_sale - total_costs,
    )


def compare_markets(quantity_quintals, options):
    if not options:
        raise ValueError("At least one market option is required.")
    results = [calculate_net_return(quantity_quintals, o) for o in options]
    return sorted(results, key=lambda r: r.net_return, reverse=True)