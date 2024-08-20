import json
from semantic_kernel.plugin_definition import kernel_function, kernel_function_context_parameter
from semantic_kernel.kernel_pydantic import KernelBaseModel
from semantic_kernel import KernelContext
from libs.utils import check_libary_major_version


check_libary_major_version('semantic_kernel', '0.5.0')


class PluginSAP(KernelBaseModel):
    account_id: str = 'not-used'

    @kernel_function(
        description="Return past purchase orders for the customer.",
        name="get_purchase_orders",
    )
    def get_purchase_orders(self, context: KernelContext) -> str:
        result = [
            {
                'deliveryDate': '2021-01-01',
                'totalAmount': 10000,
                'items': [
                    {
                        'materialId': 'IPS Natural Die Material ND1',
                        'quantity': 1,
                    },
                    {
                        'materialId': 'Untersuchungshandschuhe Nitril light lemon Gr. M',
                        'quantity': 10,
                    },
                    {
                        'materialId': 'Antiseptica r.f.u. Händedesinfektion Flasche 1 Liter',
                        'quantity': 1,
                    },
                ]
            }
        ]
        return json.dumps(result)

    @kernel_function(
        description="Create a new purchase order for the customer.",
        name="create_purchase_order",
    )
    @kernel_function_context_parameter(
        name="material_id",
        description="Name of the item to order.",
        type = "string",
        required = True,
    )
    @kernel_function_context_parameter(
        name="amound",
        description="Amount of items order.",
        type = "integer",
        required = True,
    )
    def create_purchase_order(self, context: KernelContext) -> str:
        return 'Order placed successfully'

    @kernel_function(
        description="Schedule a service appointment for the customer.",
        name="create_service_appointment",
    )
    @kernel_function_context_parameter(
        name="description",
        description="Description of the issue.",
        type = "string",
        required = True,
    )
    def create_service_appointment(self, context: KernelContext) -> str:
        return 'Appointment scheduled successfully'


# Example usage
'''
from libs.plugins.plugin_sap import PluginSAP
import libs.plugin_converter.semantic_kernel_v0_to_openai_function as semantic_kernel_v0_to_openai_function

p = PluginSAP()
print(p.get_purchase_orders(semantic_kernel_v0_to_openai_function.make_context({})))

print(p.create_purchase_order(semantic_kernel_v0_to_openai_function.make_context({
    'material_id': 'gloves',
    'quantity': 100,
})))

print(p.create_purchase_order(semantic_kernel_v0_to_openai_function.make_context({
    'description': 'Fix the broken Handstückschlauch',
})))
'''
