import logging
import re
from typing import Union, List
from urllib.parse import urlparse, unquote

import dspy
import wikipedia


def get_wiki_page_title_and_toc(url):
    """Get the main title and table of contents from a url of a Wikipedia page using the official library."""
    title = None  # Initialize title to ensure it's available for logging
    try:
        parsed_url = urlparse(url)
        # Extract language from the hostname (e.g., 'en' from 'en.wikipedia.org')
        lang = parsed_url.hostname.split('.')[0]
        # Extract title from the path, decode URL encoding, and replace underscores
        title = unquote(parsed_url.path.split('/')[-1]).replace('_', ' ')

        wikipedia.set_lang(lang)
        # Enable auto_suggest to handle redirects and close matches adaptively.
        page = wikipedia.page(title, auto_suggest=True, redirect=True)

        # The main title is simply the page's title
        main_title = page.title

        # The ToC can be constructed from the sections attribute
        # We manually add indentation to represent structure
        toc_lines = []
        for section_title in page.sections:
            toc_lines.append(f"  {section_title}") # Simulating a flat ToC structure for now

        toc = "\n".join(toc_lines)
        logging.info(f"Successfully fetched TOC for page '{main_title}' from URL {url}.")
        return main_title, toc.strip()

    except wikipedia.exceptions.PageError:
        logging.warning(f"Wikipedia page not found for title '{title}' from URL {url}. Skipping.")
        return None, None
    except wikipedia.exceptions.DisambiguationError as e:
        logging.warning(f"Disambiguation page found for URL {url}. Using first option: {e.options[0]}.")
        # Retry with the first option from the disambiguation page
        try:
            # Pass the suggested option directly to the page function
            page = wikipedia.page(e.options[0], auto_suggest=False, redirect=True)
            main_title = page.title
            toc_lines = [f"  {section_title}" for section_title in page.sections]
            toc = "\n".join(toc_lines)
            logging.info(f"Successfully fetched disambiguated page '{main_title}' from URL {url}.")
            return main_title, toc.strip()
        except Exception as inner_e:
            logging.error(f"Failed to fetch disambiguated page for {url}: {inner_e}")
            return None, None
    except Exception as e:
        logging.error(f"An unexpected error occurred while fetching from Wikipedia for URL {url}: {e}")
        return None, None


class FindRelatedTopic(dspy.Signature):
    """I'm writing a Wikipedia page for a topic mentioned below. Please identify and recommend some Wikipedia pages on closely related subjects. I'm looking for examples that provide insights into interesting aspects commonly associated with this topic, or examples that help me understand the typical content and structure included in Wikipedia pages for similar topics.
    Please list the urls in separate lines."""

    topic = dspy.InputField(prefix="Topic of interest:", format=str)
    related_topics = dspy.OutputField(format=str)


class GenPersona(dspy.Signature):
    """You need to select a group of Wikipedia editors who will work together to create a comprehensive article on the topic. Each of them represents a different perspective, role, or affiliation related to this topic. You can use other Wikipedia pages of related topics for inspiration. For each editor, add a description of what they will focus on.
    Give your answer in the following format: 1. short summary of editor 1: description\n2. short summary of editor 2: description\n...
    """

    topic = dspy.InputField(prefix="Topic of interest:", format=str)
    examples = dspy.InputField(
        prefix="Wiki page outlines of related topics for inspiration:\n", format=str
    )
    personas = dspy.OutputField(format=str)


class CreateWriterWithPersona(dspy.Module):
    """Discover different perspectives of researching the topic by reading Wikipedia pages of related topics."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.find_related_topic = dspy.ChainOfThought(FindRelatedTopic)
        self.gen_persona = dspy.ChainOfThought(GenPersona)
        self.engine = engine

    def forward(self, topic: str, draft=None):
        with dspy.settings.context(lm=self.engine):
            # Get section names from wiki pages of relevant topics for inspiration.
            related_topics = self.find_related_topic(topic=topic).related_topics
            urls = []
            for s in related_topics.split("\n"):
                if "http" in s:
                    urls.append(s[s.find("http") :])
            examples = []
            for url in urls:
                try:
                    title, toc = get_wiki_page_title_and_toc(url)
                    if title and toc:
                        examples.append(f"Title: {title}\nTable of Contents: {toc}")
                except Exception as e:
                    logging.error(f"Error occurs when processing {url}: {e}")
                    continue
            if len(examples) == 0:
                examples.append("N/A")
            gen_persona_output = self.gen_persona(
                topic=topic, examples="\n----------\n".join(examples)
            ).personas

        personas = []
        for s in gen_persona_output.split("\n"):
            match = re.search(r"\d+\.\s*(.*)", s)
            if match:
                personas.append(match.group(1))

        sorted_personas = personas

        return dspy.Prediction(
            personas=personas,
            raw_personas_output=sorted_personas,
            related_topics=related_topics,
        )


class StormPersonaGenerator:
    """
    A generator class for creating personas based on a given topic.

    This class uses an underlying engine to generate personas tailored to the specified topic.
    The generator integrates with a `CreateWriterWithPersona` instance to create diverse personas,
    including a default 'Basic fact writer' persona.

    Attributes:
        create_writer_with_persona (CreateWriterWithPersona): An instance responsible for
            generating personas based on the provided engine and topic.

    Args:
        engine (Union[dspy.dsp.LM, dspy.dsp.HFModel]): The underlying engine used for generating
            personas. It must be an instance of either `dspy.dsp.LM` or `dspy.dsp.HFModel`.
    """

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.create_writer_with_persona = CreateWriterWithPersona(engine=engine)

    def generate_persona(self, topic: str, max_num_persona: int = 3) -> List[str]:
        """
        Generates a list of personas based on the provided topic, up to a maximum number specified.

        This method first creates personas using the underlying `create_writer_with_persona` instance
        and then prepends a default 'Basic fact writer' persona to the list before returning it.
        The number of personas returned is limited to `max_num_persona`, excluding the default persona.

        Args:
            topic (str): The topic for which personas are to be generated.
            max_num_persona (int): The maximum number of personas to generate, excluding the
                default 'Basic fact writer' persona.

        Returns:
            List[str]: A list of persona descriptions, including the default 'Basic fact writer' persona
                and up to `max_num_persona` additional personas generated based on the topic.
        """
        personas = self.create_writer_with_persona(topic=topic)
        default_persona = "Basic fact writer: Basic fact writer focusing on broadly covering the basic facts about the topic."
        considered_personas = [default_persona] + personas.personas[:max_num_persona]
        return considered_personas