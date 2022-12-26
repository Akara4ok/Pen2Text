import React from 'react';
import classes from './UploadButton.scss';
import { FaFileUpload } from 'react-icons/fa';

class UploadButton extends React.Component {
    constructor(props) {
        super(props);
    }
    render() {
        return (
            <label className={classes.wrapper}>
                <FaFileUpload />
                <input
                    type="file"
                    id="fileUpload"
                    accept=".jpg, .jpeg, .png, .pdf"
                    onChange={event => this.props.uploadHandler(event)}
                    multiple
                    required
                />
                Upload
            </label>
        );
    }
}

export default UploadButton;
