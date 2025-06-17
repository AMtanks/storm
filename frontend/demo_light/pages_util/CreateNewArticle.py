import os
import time

import demo_util
import streamlit as st
from demo_util import (
    DemoFileIOHelper,
    DemoTextProcessingHelper,
    DemoUIHelper,
)


def handle_not_started():
    if st.session_state["page3_write_article_state"] == "not started":
        st.write("## Create a New Article with STORM Wiki SiliconFlow")
        
        with st.expander("Advanced Settings", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Model Settings")
                st.session_state["temperature"] = st.slider(
                    "Temperature", min_value=0.1, max_value=1.0, value=1.0, step=0.1
                )
                st.session_state["top_p"] = st.slider(
                    "Top P", min_value=0.1, max_value=1.0, value=0.9, step=0.1
                )
                
                st.write("#### Pipeline Stages")
                st.session_state["do_research"] = st.checkbox("Research Topic", value=True)
                st.session_state["do_generate_outline"] = st.checkbox("Generate Outline", value=True)
                st.session_state["do_generate_article"] = st.checkbox("Generate Article", value=True)
                st.session_state["do_polish_article"] = st.checkbox("Polish Article", value=True)
                st.session_state["remove_duplicate"] = st.checkbox("Remove Duplicate Content", value=False)
            
            with col2:
                st.write("#### Search Engine")
                st.session_state["retriever"] = st.selectbox(
                    "Search Engine",
                    options=["duckduckgo", "bing", "you", "brave", "serper", "tavily", "searxng"],
                    index=0,
                )
                
                st.write("#### Research Parameters")
                st.session_state["max_conv_turn"] = st.slider(
                    "Max Conversation Turns", min_value=1, max_value=10, value=3, step=1
                )
                st.session_state["max_perspective"] = st.slider(
                    "Max Perspectives", min_value=1, max_value=10, value=3, step=1
                )
                st.session_state["search_top_k"] = st.slider(
                    "Search Top K", min_value=1, max_value=10, value=3, step=1
                )
                st.session_state["max_thread_num"] = st.slider(
                    "Max Thread Number", min_value=1, max_value=10, value=3, step=1
                )
                
                # If DuckDuckGo is selected, show additional settings
                if st.session_state["retriever"] == "duckduckgo":
                    st.write("#### DuckDuckGo Specific Settings")
                    st.session_state["request_delay"] = st.slider(
                        "Request Delay (seconds)", min_value=0.1, max_value=5.0, value=3.0, step=0.1
                    )
                    st.session_state["max_retries"] = st.slider(
                        "Max Retries", min_value=1, max_value=10, value=5, step=1
                    )
                    st.session_state["max_delay"] = st.slider(
                        "Max Delay (seconds)", min_value=1.0, max_value=15.0, value=8.0, step=0.5
                    )
                    st.session_state["use_multiple_backends"] = st.checkbox("Use Multiple Backends", value=False)
                    st.session_state["exponential_backoff"] = st.checkbox("Use Exponential Backoff", value=True)
                    st.session_state["webpage_helper_max_threads"] = st.slider(
                        "Webpage Helper Max Threads", min_value=1, max_value=5, value=1, step=1
                    )
        
        _, search_form_column, _ = st.columns([2, 5, 2])
        with search_form_column:
            with st.form(key="search_form"):
                # Text input for the search topic
                DemoUIHelper.st_markdown_adjust_size(
                    content="Enter the topic you want to learn in depth:", font_size=18
                )
                st.session_state["page3_topic"] = st.text_input(
                    label="page3_topic", label_visibility="collapsed"
                )
                pass_appropriateness_check = True

                # Submit button for the form
                submit_button = st.form_submit_button(label="Research")
                # only start new search when button is clicked, not started, or already finished previous one
                if submit_button and st.session_state["page3_write_article_state"] in [
                    "not started",
                    "show results",
                ]:
                    if not st.session_state["page3_topic"].strip():
                        pass_appropriateness_check = False
                        st.session_state["page3_warning_message"] = (
                            "topic could not be empty"
                        )

                    st.session_state["page3_topic_name_cleaned"] = (
                        st.session_state["page3_topic"]
                        .replace(" ", "_")
                        .replace("/", "_")
                    )
                    st.session_state["page3_topic_name_truncated"] = demo_util.truncate_filename(
                        st.session_state["page3_topic_name_cleaned"]
                    )
                    if not pass_appropriateness_check:
                        st.session_state["page3_write_article_state"] = "not started"
                        alert = st.warning(
                            st.session_state["page3_warning_message"], icon="⚠️"
                        )
                        time.sleep(5)
                        alert.empty()
                    else:
                        st.session_state["page3_write_article_state"] = "initiated"


def handle_initiated():
    if st.session_state["page3_write_article_state"] == "initiated":
        current_working_dir = os.path.join(demo_util.get_demo_dir(), "DEMO_WORKING_DIR")
        if not os.path.exists(current_working_dir):
            os.makedirs(current_working_dir)

        if "runner" not in st.session_state:
            demo_util.set_storm_runner()
        st.session_state["page3_current_working_dir"] = current_working_dir
        st.session_state["page3_write_article_state"] = "pre_writing"


def handle_pre_writing():
    if st.session_state["page3_write_article_state"] == "pre_writing":
        status = st.status(
            "I am brain**STORM**ing now to research the topic. (This may take 2-3 minutes.)"
        )
        st_callback_handler = demo_util.StreamlitCallbackHandler(status)
        with status:
            # Check if the topic contains Chinese characters and translate if needed
            original_topic = st.session_state["page3_topic"]
            english_topic, is_translated = demo_util.translate_topic_to_english(original_topic)
            
            if is_translated:
                st.info(f"Translated topic: {original_topic} → {english_topic}")
                # Save the translation information
                translation_dir = os.path.join(
                    st.session_state["page3_current_working_dir"],
                    st.session_state["page3_topic_name_truncated"],
                )
                os.makedirs(translation_dir, exist_ok=True)
                with open(os.path.join(translation_dir, "topic_translation.txt"), "w", encoding="utf-8") as f:
                    f.write(f"原始主题: {original_topic}\n")
                    f.write(f"英文主题: {english_topic}\n")
                
                # Use the translated topic for search
                search_topic = english_topic
            else:
                search_topic = original_topic
            
            # STORM main gen outline
            st.session_state["runner"].run(
                topic=search_topic,
                do_research=st.session_state.get("do_research", True),
                do_generate_outline=st.session_state.get("do_generate_outline", True),
                do_generate_article=False,
                do_polish_article=False,
                callback_handler=st_callback_handler,
                original_topic=original_topic,  # Pass original topic for display purposes
            )
            conversation_log_path = os.path.join(
                st.session_state["page3_current_working_dir"],
                st.session_state["page3_topic_name_truncated"],
                "conversation_log.json",
            )
            demo_util._display_persona_conversations(
                DemoFileIOHelper.read_json_file(conversation_log_path)
            )
            st.session_state["page3_write_article_state"] = "final_writing"
            status.update(label="brain**STORM**ing complete!", state="complete")


def handle_final_writing():
    if st.session_state["page3_write_article_state"] == "final_writing":
        # polish final article
        with st.status(
            "Now I will connect the information I found for your reference. (This may take 4-5 minutes.)"
        ) as status:
            st.info(
                "Now I will connect the information I found for your reference. (This may take 4-5 minutes.)"
            )
            
            # Get the original topic and its translated version if it exists
            original_topic = st.session_state["page3_topic"]
            translation_file = os.path.join(
                st.session_state["page3_current_working_dir"],
                st.session_state["page3_topic_name_truncated"],
                "topic_translation.txt"
            )
            
            # Determine which topic to use for generation
            search_topic = original_topic
            if os.path.exists(translation_file):
                with open(translation_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    if len(lines) >= 2:
                        english_topic = lines[1].strip().replace("英文主题: ", "")
                        search_topic = english_topic
            
            st.session_state["runner"].run(
                topic=search_topic,
                do_research=False,
                do_generate_outline=False,
                do_generate_article=st.session_state.get("do_generate_article", True),
                do_polish_article=st.session_state.get("do_polish_article", True),
                remove_duplicate=st.session_state.get("remove_duplicate", False),
                original_topic=original_topic,  # Pass original topic for display purposes
            )
            # finish the session
            st.session_state["runner"].post_run()

            # update status bar
            st.session_state["page3_write_article_state"] = "prepare_to_show_result"
            status.update(label="information snythesis complete!", state="complete")


def handle_prepare_to_show_result():
    if st.session_state["page3_write_article_state"] == "prepare_to_show_result":
        _, show_result_col, _ = st.columns([4, 3, 4])
        with show_result_col:
            if st.button("show final article"):
                st.session_state["page3_write_article_state"] = "completed"
                st.rerun()


def handle_completed():
    if st.session_state["page3_write_article_state"] == "completed":
        # display polished article
        current_working_dir_paths = DemoFileIOHelper.read_structure_to_dict(
            st.session_state["page3_current_working_dir"]
        )
        current_article_file_path_dict = current_working_dir_paths[
            st.session_state["page3_topic_name_truncated"]
        ]
        demo_util.display_article_page(
            selected_article_name=st.session_state["page3_topic_name_cleaned"],
            selected_article_file_path_dict=current_article_file_path_dict,
            show_title=True,
            show_main_article=True,
        )


def create_new_article_page():
    demo_util.clear_other_page_session_state(page_index=3)

    if "page3_write_article_state" not in st.session_state:
        st.session_state["page3_write_article_state"] = "not started"

    handle_not_started()

    handle_initiated()

    handle_pre_writing()

    handle_final_writing()

    handle_prepare_to_show_result()

    handle_completed()
