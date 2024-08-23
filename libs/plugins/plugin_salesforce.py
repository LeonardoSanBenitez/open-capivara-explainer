import json
from semantic_kernel.plugin_definition import kernel_function, kernel_function_context_parameter
from semantic_kernel.kernel_pydantic import KernelBaseModel
from semantic_kernel import KernelContext
from libs.utils import check_libary_major_version


check_libary_major_version('semantic_kernel', '0.5.0')


class PluginSalesforce(KernelBaseModel):
    account_id: str = 'not-used'

    @kernel_function(
        description="Return more information about the customer (name, etc).",
        name="get_account_details",
    )
    def get_account_details(self, context: KernelContext) -> str:
        result = {
            'name': 'John Doe',
            'email': 'john.d@something.com',
            'phone': '123-456-7890',
            'address': '1234 Elm St, Springfield, IL 62701',
        }
        return json.dumps(result)

    @kernel_function(
        description="Return the latests sales opportunities for the customer.",
        name="get_opportunity",
    )
    def get_opportunity(self, context: KernelContext) -> str:
        result = {
            'name': 'Opportunity 1',
            'stage': 'Prospecting',
            'probability': 10,
            'amount': 10000,
            'open_date': '2021-01-01',
            'close_date': None,

        }
        return json.dumps(result)


# Example usage
'''
from libs.plugins.plugin_salesforce import PluginSalesforce
import libs.plugin_converter.semantic_kernel_v0_to_openai_function as semantic_kernel_v0_to_openai_function

p = PluginSalesforce()
print(p.get_account_details(semantic_kernel_v0_to_openai_function.make_context({})))
print(p.get_opportunity(semantic_kernel_v0_to_openai_function.make_context({})))
'''
