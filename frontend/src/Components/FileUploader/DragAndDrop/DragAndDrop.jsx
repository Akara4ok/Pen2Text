import React from 'react';
import classes from './DragAndDrop.scss';

class DragAndDrop extends React.Component {
    constructor(props) {
        super(props);
    }

    dropHandler = event => {
        event.preventDefault();

        const files = [];
        if (event.dataTransfer.items) {
            [...event.dataTransfer.items].forEach((item, i) => {
                if (item.kind === 'file') {
                    const file = item.getAsFile();
                    files.push(file);
                }
            });
        } else {
            [...event.dataTransfer.files].forEach((file, i) => {
                files.push(file);
            });
        }
        this.props.onDrop(files);
    };

    dragOverHandler = event => {
        event.preventDefault();
    };

    dragStartHandler = () => {
        console.log('drag');
    };

    render() {
        const { className } = this.props;
        return (
            <div
                className={`${classes.wrapper} ${className ?? ''}`}
                onDrop={this.dropHandler}
                onDragOver={this.dragOverHandler}
                onDragStart={this.dragStartHandler}>
                <p>Drop to upload files</p>
            </div>
        );
    }
}

export default DragAndDrop;
