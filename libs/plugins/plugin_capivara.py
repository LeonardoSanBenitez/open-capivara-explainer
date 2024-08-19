import requests
from semantic_kernel.plugin_definition import kernel_function, kernel_function_context_parameter
from semantic_kernel.kernel_pydantic import KernelBaseModel
from semantic_kernel import KernelContext
from libs.utils import check_libary_major_version


check_libary_major_version('semantic_kernel', '0.5.0')

class PluginCapivaraAlert(KernelBaseModel):
    capivara_base_url: str
    alert_id: str

    @kernel_function(
        description="Close the current ticket. Only call this function is explicitly requested by the user.",
        name="close_ticket",
    )
    def close_ticket(self, context: KernelContext) -> str:
        response = requests.post('http://capivara:80/v1/public-api-update-incidents', json=[
            {
                "id": self.alert_id,
                "state": "closed",
            }
        ])
        assert response.status_code == 200
        assert all([r[0] == 200 for r in response.json()])
        return 'Ticket closed successfully'