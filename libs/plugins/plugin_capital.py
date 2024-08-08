from semantic_kernel.plugin_definition import kernel_function, kernel_function_context_parameter
from semantic_kernel.kernel_pydantic import KernelBaseModel
from semantic_kernel import KernelContext


###############################
# Version check
import importlib.metadata
from packaging import version
required_major_version = '0.5.0'
installed_version = importlib.metadata.version('semantic_kernel')

# Parse the version to handle semantic versioning properly
installed_major_version = version.parse(installed_version).base_version

if installed_major_version != required_major_version:
    raise ImportError(f"semantic_kernel version {required_major_version} is required, but {installed_version} is installed.")
# End of version check
###############################


country_to_capital = {
    "afghanistan": "kabul",
    "albania": "tirana",
    "algeria": "algiers",
    "andorra": "andorra la vella",
    "angola": "luanda",
    "argentina": "buenos aires",
    "armenia": "yerevan",
    "australia": "canberra",
    "austria": "vienna",
    "azerbaijan": "baku",
    "bahamas": "nassau",
    "bahrain": "manama",
    "bangladesh": "dhaka",
    "barbados": "bridgetown",
    "belarus": "minsk",
    "belgium": "brussels",
    "belize": "belmopan",
    "benin": "porto-novo",
    "bhutan": "thimphu",
    "bolivia": "sucre",
    "bosnia and herzegovina": "sarajevo",
    "botswana": "gaborone",
    "brazil": "brasília",
    "brunei": "bandar seri begawan",
    "bulgaria": "sofia",
    "burkina faso": "ouagadougou",
    "burundi": "gitega",
    "cambodia": "phnom penh",
    "cameroon": "yaoundé",
    "canada": "ottawa",
    "cape verde": "praia",
    "central african republic": "bangui",
    "chad": "n'djamena",
    "chile": "santiago",
    "china": "beijing",
    "colombia": "bogotá",
    "comoros": "moroni",
    "congo, democratic republic of the": "kinshasa",
    "congo, republic of the": "brazzaville",
    "costa rica": "san josé",
    "croatia": "zagreb",
    "cuba": "havana",
    "cyprus": "nicosia",
    "czech republic": "prague",
    "denmark": "copenhagen",
    "djibouti": "djibouti",
    "dominica": "roseau",
    "dominican republic": "santo domingo",
    "east timor": "dili",
    "ecuador": "quito",
    "egypt": "cairo",
    "el salvador": "san salvador",
    "equatorial guinea": "malabo",
    "eritrea": "asmara",
    "estonia": "tallinn",
    "eswatini": "mbabane",
    "ethiopia": "addis ababa",
    "fiji": "suva",
    "finland": "helsinki",
    "france": "paris",
    "gabon": "libreville",
    "gambia": "banjul",
    "georgia": "tbilisi",
    "germany": "berlin",
    "ghana": "accra",
    "greece": "athens",
    "grenada": "st. george's",
    "guatemala": "guatemala city",
    "guinea": "conakry",
    "guinea-bissau": "bissau",
    "guyana": "georgetown",
    "haiti": "port-au-prince",
    "honduras": "tegucigalpa",
    "hungary": "budapest",
    "iceland": "reykjavik",
    "india": "new delhi",
    "indonesia": "jakarta",
    "iran": "tehran",
    "iraq": "baghdad",
    "ireland": "dublin",
    "israel": "jerusalem",
    "italy": "rome",
    "ivory coast": "yamoussoukro",
    "jamaica": "kingston",
    "japan": "tokyo",
    "jordan": "amman",
    "kazakhstan": "astana",
    "kenya": "nairobi",
    "kiribati": "tarawa",
    "korea, north": "pyongyang",
    "korea, south": "seoul",
    "kosovo": "pristina",
    "kuwait": "kuwait city",
    "kyrgyzstan": "bishkek",
    "laos": "vientiane",
    "latvia": "riga",
    "lebanon": "beirut",
    "lesotho": "maseru",
    "liberia": "monrovia",
    "libya": "tripoli",
    "liechtenstein": "vaduz",
    "lithuania": "vilnius",
    "luxembourg": "luxembourg",
    "madagascar": "antananarivo",
    "malawi": "lilongwe",
    "malaysia": "kuala lumpur",
    "maldives": "malé",
    "mali": "bamako",
    "malta": "valletta",
    "marshall islands": "majuro",
    "mauritania": "nouakchott",
    "mauritius": "port louis",
    "mexico": "mexico city",
    "micronesia": "palikir",
    "moldova": "chișinău",
    "monaco": "monaco",
    "mongolia": "ulaanbaatar",
    "montenegro": "podgorica",
    "morocco": "rabat",
    "mozambique": "maputo",
    "myanmar": "naypyidaw",
    "namibia": "windhoek",
    "nauru": "yaren",
    "nepal": "kathmandu",
    "netherlands": "amsterdam",
    "new zealand": "wellington",
    "nicaragua": "managua",
    "niger": "niamey",
    "nigeria": "abuja",
    "north macedonia": "skopje",
    "norway": "oslo",
    "oman": "muscat",
    "pakistan": "islamabad",
    "palau": "ngerulmud",
    "palestine": "ramallah",
    "panama": "panama city",
    "papua new guinea": "port moresby",
    "paraguay": "asunción",
    "peru": "lima",
    "philippines": "manila",
    "poland": "warsaw",
    "portugal": "lisbon",
    "qatar": "doha",
    "romania": "bucharest",
    "russia": "moscow",
    "rwanda": "kigali",
    "saint kitts and nevis": "basseterre",
    "saint lucia": "castries",
    "saint vincent and the grenadines": "kingstown",
    "samoa": "apia",
    "san marino": "san marino",
    "sao tome and principe": "sao tome",
    "saudi arabia": "riyadh",
    "senegal": "dakar",
    "serbia": "belgrade",
    "seychelles": "victoria",
    "sierra leone": "freetown",
    "singapore": "singapore",
    "slovakia": "bratislava",
    "slovenia": "ljubljana",
    "solomon islands": "honiara",
    "somalia": "mogadishu",
    "south africa": "pretoria",
    "south sudan": "juba",
    "spain": "madrid",
    "sri lanka": "colombo",
    "sudan": "khartoum",
    "suriname": "paramaribo",
    "sweden": "stockholm",
    "switzerland": "bern",
    "syria": "damascus",
    "taiwan": "taipei",
    "tajikistan": "dushanbe",
    "tanzania": "dodoma",
    "thailand": "bangkok",
    "togo": "lomé",
    "tonga": "nukuʻalofa",
    "trinidad and tobago": "port of spain",
    "tunisia": "tunis",
    "turkey": "ankara",
    "turkmenistan": "ashgabat",
    "tuvalu": "funafuti",
    "uganda": "kampala",
    "ukraine": "kyiv",
    "united arab emirates": "abu dhabi",
    "united kingdom": "london",
    "united states": "washington, d.c.",
    "uruguay": "montevideo",
    "uzbekistan": "tashkent",
    "vanuatu": "port vila",
    "vatican city": "vatican city",
    "venezuela": "caracas",
    "vietnam": "hanoi",
    "yemen": "sana'a",
    "zambia": "lusaka",
    "zimbabwe": "harare"
}


class PluginCapital(KernelBaseModel):
    @kernel_function(
        description="Returns the name of the capital of a country.",
        name="get_capital",
    )
    @kernel_function_context_parameter(
        name="country",
        description="Name of the country.",
        type = "string",
        required = True,
    )
    def get_capital(self, context: KernelContext) -> str:
        # Parameter validation
        if 'country' not in context.variables:
            raise ValueError('Missing parameter country')
        if type(context.variables['country']) != str:
            raise ValueError('Parameter entity_name should be a string')
        country = context.variables['country']
        country = country.replace('`', '').replace('"', '').replace("'", '')
        country = country.strip()
        country = country.lower()

        # Execute
        if country in country_to_capital:
            return country_to_capital[country]
        else:
            raise RuntimeError('Country not found')


# Usage
'''
from libs.plugins.plugin_capital import PluginCapital
from libs.plugin_converter.semantic_kernel_v0_to_openai_tool import make_context
plugin = PluginCapital()
print(plugin.get_capital(make_context({'country': 'Brazil'})))
'''
